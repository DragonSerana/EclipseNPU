// R3.1 激励：DMA 单体（DMA_LOAD，DDR → SRAM）
//
// 要证的事：
//   1. SRAM 侧写出来的内容，和 cmodel 的 memcpy 逐 bit 一致
//   2. 只写"落在本行范围内"的字，行间空隙不写
//   3. RTL 数出来的突发数 == cmodel 的公式
//   4. RTL 发请求的拍数 == cmodel 的 ceil(bursts/2)
//   5. 剩下的就是 DMA_FIXED_OVERHEAD，看它由哪些拍组成

#include "Vdma.h"
#include "verilated.h"
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

static const int      NBANK       = 32;
static const int      AW          = 12;
static const int      DDR_LATENCY = 4;
static const uint32_t DDR_SIZE    = 1u << 20;   // 1MB
static const uint32_t SRAM_WORDS  = 1u << AW;   // 4096 字 = 8KB
static const uint16_t SENTINEL    = 0xFFFF;

static Vdma               *top;
static std::vector<uint8_t>  ddr;
static std::vector<uint16_t> sram_img;
static std::vector<uint16_t> sram_exp;

static uint64_t cycle_cnt;

static void tick() { top->clk = 0; top->eval(); }
static void edge() { top->clk = 1; top->eval(); }

// ---------------------------------------------------------------- 行为级 DDR
struct Slot { bool v; uint32_t addr; };
static Slot ring[2][DDR_LATENCY];
static int  ring_p;

static void set_rdata(int k, uint32_t addr) {
  for (int w = 0; w < 4; w++) {
    uint32_t v = 0;
    for (int b = 0; b < 4; b++)
      v |= (uint32_t)ddr[(addr + w * 4 + b) & (DDR_SIZE - 1)] << (8 * b);
    top->ddr_rdata[k][w] = v;
  }
}

static void ddr_reset() {
  for (int k = 0; k < 2; k++)
    for (int i = 0; i < DDR_LATENCY; i++) { ring[k][i].v = false; ring[k][i].addr = 0; }
  ring_p = 0;
  top->ddr_rvalid[0] = 0;
  top->ddr_rvalid[1] = 0;
  for (int k = 0; k < 2; k++)
    for (int w = 0; w < 4; w++) top->ddr_rdata[k][w] = 0;
}

static bool req0, req1;

// 走一拍。顺序：clk=0 求值 -> 采样请求 -> 驱动 DDR 返回 -> 重新求值
// -> 采样写口 -> 本拍请求入环 -> clk=1 上升沿
static void step() {
  tick();

  req0 = top->ddr_ren[0];
  req1 = top->ddr_ren[1];
  uint32_t a0 = top->ddr_raddr[0];
  uint32_t a1 = top->ddr_raddr[1];

  Slot s0 = ring[0][ring_p];
  Slot s1 = ring[1][ring_p];
  top->ddr_rvalid[0] = s0.v;
  top->ddr_rvalid[1] = s1.v;
  set_rdata(0, s0.addr);
  set_rdata(1, s1.addr);
  top->eval();

  for (int k = 0; k < NBANK; k++)
    if (top->wr_valid[k])
      sram_img[top->wr_addr[k] & (SRAM_WORDS - 1)] = top->wr_data[k];

  ring[0][ring_p].v = req0;
  ring[0][ring_p].addr = a0;
  ring[1][ring_p].v = req1;
  ring[1][ring_p].addr = a1;
  ring_p = (ring_p + 1) % DDR_LATENCY;

  edge();
  cycle_cnt++;
}

// ---------------------------------------------------------------- 参考模型
struct Desc {
  const char *name;
  uint32_t    sram_addr, ddr_addr, rows, cols, stride;
};

static uint64_t ref_bursts(const Desc &d) {
  const uint32_t rowBytes = d.cols * 2;
  uint64_t b = 0;
  for (uint32_t i = 0; i < d.rows; i++) {
    const uint32_t start = d.ddr_addr + d.stride * i;
    b += (start + rowBytes - 1) / 16 - start / 16 + 1;
  }
  return b;
}

static void ref_sram(const Desc &d) {
  for (uint32_t i = 0; i < d.rows; i++) {
    const uint8_t *src = &ddr[d.ddr_addr + d.stride * i];
    const uint32_t base = (d.sram_addr >> 1) + i * d.cols;   // dstStride = cols*2
    for (uint32_t j = 0; j < d.cols; j++)
      sram_exp[(base + j) & (SRAM_WORDS - 1)] =
          (uint16_t)(src[j * 2] | (src[j * 2 + 1] << 8));
  }
}

// ---------------------------------------------------------------- 跑一个案例
static bool run_case(const Desc &d) {
  std::fill(sram_img.begin(), sram_img.end(), SENTINEL);
  std::fill(sram_exp.begin(), sram_exp.end(), SENTINEL);
  ref_sram(d);

  top->rst_n = 0;
  top->desc_start = 0;
  top->desc_sram_addr  = d.sram_addr;
  top->desc_ddr_addr   = d.ddr_addr;
  top->desc_rows       = d.rows;
  top->desc_cols       = d.cols;
  top->desc_src_stride = d.stride;
  ddr_reset();
  cycle_cnt = 0;

  tick(); edge();
  tick(); edge();
  top->rst_n = 1;

  top->desc_start = 1;
  step();
  top->desc_start = 0;

  uint64_t n_bursts = 0, n_issue_cyc = 0, t_first = 0, t_last = 0;
  bool     first_seen = false;
  uint64_t guard = 0;

  while (true) {
    step();
    if (req0) n_bursts++;
    if (req1) n_bursts++;
    if (req0 || req1) {
      n_issue_cyc++;
      if (!first_seen) { first_seen = true; t_first = cycle_cnt; }
      t_last = cycle_cnt;
    }
    if (top->done) break;
    if (++guard > 200000) { std::printf("  !! 跑飞了\n"); return false; }
  }

  const uint64_t bs = ref_bursts(d);
  const uint64_t cs = (bs * 16 + 31) / 32;   // cmodel: ceil(bursts*16/32)

  bool ok = true;
  int  n_bad = 0;
  for (uint32_t w = 0; w < SRAM_WORDS; w++) {
    if (sram_img[w] != sram_exp[w]) {
      ok = false;
      if (n_bad < 6)
        std::printf("  !! 字地址 %4u（bank %2u local %3u）= 0x%04x，期望 0x%04x\n",
                    w, w % NBANK, w / NBANK, sram_img[w], sram_exp[w]);
      n_bad++;
    }
  }

  std::printf("\n【%s】rows=%u cols=%u stride=%u\n", d.name, d.rows, d.cols, d.stride);
  std::printf("  内容    ：%s", ok ? "逐 bit 一致" : "不符，见上");
  if (!ok) std::printf("（%d 个字不同）", n_bad);
  std::printf("\n");
  std::printf("  突发数  ：RTL %llu  /  模型 %llu  %s\n",
              (unsigned long long)n_bursts, (unsigned long long)bs,
              n_bursts == bs ? "一致" : "<<< 不符");
  std::printf("  搬运拍数：RTL %llu（发请求 %llu 拍，首 %llu 末 %llu） /  模型 %llu  %s\n",
              (unsigned long long)(n_issue_cyc), (unsigned long long)n_issue_cyc,
              (unsigned long long)t_first, (unsigned long long)t_last,
              (unsigned long long)cs, n_issue_cyc == cs ? "一致" : "<<< 不符");
  std::printf("  总拍数  ：RTL %llu（start→done） /  模型 %llu（搬运+16）\n",
              (unsigned long long)cycle_cnt, (unsigned long long)(cs + 16));

  return ok && n_bursts == bs;
}

int main(int argc, char **argv) {
  Verilated::commandArgs(argc, argv);
  top = new Vdma;

  ddr.resize(DDR_SIZE);
  for (uint32_t i = 0; i < DDR_SIZE; i++) ddr[i] = (uint8_t)(i * 31 + 7);
  sram_img.assign(SRAM_WORDS, SENTINEL);
  sram_exp.assign(SRAM_WORDS, SENTINEL);

  top->rst_n = 0;
  top->desc_start = 0;
  top->desc_sram_addr = 0;
  top->desc_ddr_addr = 0;
  top->desc_rows = 0;
  top->desc_cols = 0;
  top->desc_src_stride = 0;
  ddr_reset();

  static const Desc cases[] = {
      {"golden A   A 的 K 块 0", 0x000, 0x01000, 128, 16, 256},
      {"golden B   B 的 K 块 0", 0x800, 0x01000, 16, 128, 256},
      {"isa       loadMatmulLhs", 0x100, 0x08000, 15, 31, 64},
      {"isa       storeActDst", 0x200, 0x09000, 15, 63, 126},
  };

  bool allPass = true;
  for (const Desc &d : cases)
    if (!run_case(d)) allPass = false;

  std::printf("\n%s\n", allPass ? "Pass" : "Failed");
  delete top;
  return allPass ? 0 : 1;
}
