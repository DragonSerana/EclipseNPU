#include "Vbank_xbar.h"
#include "verilated.h"
#include <cstdio>

static const int NBANK = 32;
static const int AW    = 12;

static Vbank_xbar *top;

static void tick() { top->clk = 0; top->eval(); }
static void edge() { top->clk = 1; top->eval(); }

static bool allPass = true;

static void wr_one(int addr, int data) {
  top->wr_valid[0] = 1;
  top->wr_addr[0]  = addr;
  top->wr_data[0]  = data;
  tick();
  edge();
  top->wr_valid[0] = 0;
}

// 把 a[] 里的 n 个请求全部 grant 掉，返回用了多少拍；counts 传出第一拍的 bank_cnt
static int run_reads(int *a, int n, int hash_en, int *counts) {
  top->hash_en = hash_en;
  int cycles = 0;
  while (n > 0) {
    for (int i = 0; i < NBANK; i++) top->req_addr[i] = (i < n) ? a[i] : 0;
    tick();
    if (cycles == 0)
      for (int b = 0; b < NBANK; b++) counts[b] = (int)top->bank_cnt[b];
    edge();
    int keep = 0;
    for (int i = 0; i < n; i++)
      if (!top->grant[i]) a[keep++] = a[i];
    n = keep;
    cycles++;
  }
  top->hash_en = 0;
  return cycles;
}

// A 的 nA 个请求用行距 strideA，后面跟 B 的 nB 个连续地址
static void build(int *a, int strideA, int nA, int nB) {
  for (int i = 0; i < nA; i++) a[i] = i * strideA;
  for (int j = 0; j < nB; j++) a[nA + j] = j;
}

static int scenario(const char *label, int strideA, int nA, int nB, int hash_en,
                    int expect) {
  int a[64], counts[NBANK];
  build(a, strideA, nA, nB);
  int cyc = run_reads(a, nA + nB, hash_en, counts);

  int worst = 0;
  bool full = (nA + nB == NBANK);
  if (full)
    for (int b = 0; b < NBANK; b++)
      if (counts[b] > worst) worst = counts[b];

  bool ok = (cyc == expect);
  if (!ok) allPass = false;
  if (full)
    std::printf("  %-26s 最挤 bank %2d 个请求 -> %2d 拍   %s\n", label, worst, cyc,
                ok ? "" : "<<< 与手算不符");
  else
    std::printf("  %-26s （只发 %d 个，未填满 32 槽）-> %2d 拍   %s\n", label,
                nA + nB, cyc, ok ? "" : "<<< 与手算不符");
  return cyc;
}

int main(int argc, char **argv) {
  Verilated::commandArgs(argc, argv);
  top = new Vbank_xbar;

  top->rst_n   = 0;
  top->hash_en = 0;
  for (int i = 0; i < NBANK; i++) {
    top->req_addr[i]  = 0;
    top->wr_valid[i]  = 0;
    top->wr_addr[i]   = 0;
    top->wr_data[i]   = 0;
  }
  tick(); edge(); tick(); edge();
  top->rst_n = 1;

  // 每个 bank 的本地地址 0 写 0x1000+b：全局地址 = b
  for (int b = 0; b < NBANK; b++) wr_one(b, 0x1000 + b);

  // ---- 测试 1：无冲突读，验证数据通路 ----
  for (int i = 0; i < NBANK; i++) top->req_addr[i] = i;
  tick(); edge();
  bool ok1 = true;
  for (int i = 0; i < NBANK; i++)
    if (top->grant[i] != 1 || (int)top->rd_data[i] != 0x1000 + i) {
      ok1 = false;
      allPass = false;
      std::printf("  !! slot %2d grant=%d data=0x%04x 期望 0x%04x\n", i,
                  (int)top->grant[i], (int)top->rd_data[i], 0x1000 + i);
    }
  std::printf("测试 1（无冲突读）-> %s\n\n", ok1 ? "数据全对" : "有错（见上）");

  // ---- R2.4：三种映射对照 ----
  // A 的 16 个地址 = i*stride；B 的 16 个地址 = j
  std::printf("R2.4 对照（A 的 16 个读请求 + 可选 B 的 16 个）\n");

  std::printf(" 只有 A（16 个请求，行距 %d 或 %d）\n", 128, 129);
  scenario("行主序 %32（baseline）", 128, 16, 0, 0, 16);
  scenario("padding 行距 129",      129, 16, 0, 0, 1);
  scenario("XOR 散列",              128, 16, 0, 1, 2);

  std::printf(" A + B（32 个请求，就是 R2 那个真实场景）\n");
  scenario("行主序 %32（baseline）", 128, 16, 16, 0, 17);
  scenario("padding 行距 129",      129, 16, 16, 0, 2);
  scenario("XOR 散列",              128, 16, 16, 1, 3);

  // ---- 测试 3：写侧同样的跨步地址 ----
  int pending[32];
  for (int i = 0; i < 16; i++) {
    pending[i]       = i * 128;
    top->wr_valid[i] = 1;
    top->wr_addr[i]  = i * 128;
    top->wr_data[i]  = 0xB000 + i;
  }
  int n = 16, cycles = 0, wcnt[NBANK];
  while (n > 0) {
    for (int i = 0; i < NBANK; i++) {
      top->wr_valid[i] = (i < n) ? 1 : 0;
      top->wr_addr[i]  = (i < n) ? pending[i] : 0;
    }
    tick();
    if (cycles == 0)
      for (int b = 0; b < NBANK; b++) wcnt[b] = (int)top->wr_bank_cnt[b];
    int keep = 0;
    for (int i = 0; i < n; i++)
      if (!top->wr_grant[i]) pending[keep++] = pending[i];
    n = keep;
    cycles++;
    edge();
  }
  for (int i = 0; i < NBANK; i++) top->wr_valid[i] = 0;

  std::printf("\n测试 3（写侧：16 个地址全撞 bank 0）\n");
  std::printf("  最挤 bank %d 个请求 -> %d 拍（手算 16）-> %s\n", wcnt[0], cycles,
              cycles == 16 ? "对" : "不符");
  if (cycles != 16) allPass = false;

  std::printf("\n%s\n", allPass ? "Pass" : "Failed");
  delete top;
  return allPass ? 0 : 1;
}
