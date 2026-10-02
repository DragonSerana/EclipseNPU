// R2.2b 激励：32 块 bank 并排（bank_mem），读侧写侧都 bank 化
//
// 要证的事：
//   1. 一拍能同时写全部 32 个 bank（每块地址、数据都不同）
//   2. 数据不会串 bank
//   3. bank 内地址真的在用（同一个 bank 的不同格子互不覆盖）
//   4. 单个 wr_en 能只写一块，别的 bank 不受影响
//   5. 没写过的格子读出来是 0
//
// 读是同步读（R2.1 的结论），所以 rd_data 比 rd_addr 晚一拍：
//     摆好 rd_addr -> tick -> edge（沿上采样）-> 沿后立刻读 rd_data

#include "Vbank_mem.h"
#include "verilated.h"
#include <cstdio>

static const int NBANK = 32;
static const int DEPTH = 128;

static Vbank_mem *top;

static void tick() { top->clk = 0; top->eval(); }
static void edge() { top->clk = 1; top->eval(); }

// 把 32 个读地址都设成 off，过一个沿，然后逐项对期望
static bool read_and_check(int off, int base) {
  for (int b = 0; b < NBANK; b++) top->rd_addr[b] = off;
  tick();
  edge();
  bool ok = true;
  for (int b = 0; b < NBANK; b++) {
    int exp = (base < 0) ? 0 : base + b;
    if ((int)top->rd_data[b] != exp) {
      ok = false;
      std::printf("  !! bank %2d = 0x%04x，期望 0x%04x\n", b,
                  (int)top->rd_data[b], exp);
    }
  }
  return ok;
}

int main(int argc, char **argv) {
  Verilated::commandArgs(argc, argv);
  top = new Vbank_mem;

  top->rst_n = 0;
  for (int b = 0; b < NBANK; b++) {
    top->rd_addr[b] = 0;
    top->wr_en[b] = 0;
    top->wr_addr[b] = 0;
    top->wr_data[b] = 0;
  }
  tick();
  edge();
  tick();
  edge();
  top->rst_n = 1;

  bool allPass = true;

  // ---- 阶段 1：一拍并行写全部 32 个 bank，本地地址 b+1 ----
  for (int b = 0; b < NBANK; b++) {
    top->wr_en[b] = 1;
    top->wr_addr[b] = b + 1;
    top->wr_data[b] = 0x1000 + b;
  }
  tick();
  edge(); // ← 一拍写完 32 个

  // ---- 阶段 2：一拍并行写全部 32 个 bank，本地地址 b+65 ----
  for (int b = 0; b < NBANK; b++) {
    top->wr_addr[b] = b + 65;
    top->wr_data[b] = 0x2000 + b;
  }
  tick();
  edge(); // ← 又一拍写完 32 个

  for (int b = 0; b < NBANK; b++) top->wr_en[b] = 0; // 停写

  // ---- 阶段 3：只写 bank 7（本地地址 100），验证单个 wr_en ----
  top->wr_en[7] = 1;
  top->wr_addr[7] = 100;
  top->wr_data[7] = 0xDEAD;
  tick();
  edge();
  top->wr_en[7] = 0;

  // ---- 阶段 4：每个 bank 读自己的本地地址 b+1，期望 0x1000+b ----
  bool ok4 = true;
  for (int b = 0; b < NBANK; b++) top->rd_addr[b] = b + 1;
  tick();
  edge();
  for (int b = 0; b < NBANK; b++) {
    int exp = 0x1000 + b;
    if ((int)top->rd_data[b] != exp) {
      ok4 = false;
      allPass = false;
      std::printf("  !! 阶段4 bank %2d = 0x%04x，期望 0x%04x\n", b,
                  (int)top->rd_data[b], exp);
    }
  }
  std::printf("阶段 4：读【本地地址 b+1】-> %s\n",
              ok4 ? "32 个 bank 全对" : "有错（见上）");

  // ---- 阶段 5：每个 bank 读本地地址 b+65，期望 0x2000+b ----
  bool ok5 = true;
  for (int b = 0; b < NBANK; b++) top->rd_addr[b] = b + 65;
  tick();
  edge();
  for (int b = 0; b < NBANK; b++) {
    int exp = 0x2000 + b;
    if ((int)top->rd_data[b] != exp) {
      ok5 = false;
      allPass = false;
      std::printf("  !! 阶段5 bank %2d = 0x%04x，期望 0x%04x\n", b,
                  (int)top->rd_data[b], exp);
    }
  }
  std::printf("阶段 5：读【本地地址 b+65】-> %s\n",
              ok5 ? "32 个 bank 全对" : "有错（见上）");

  // ---- 阶段 6：全部读本地地址 100，只有 bank 7 写过 ----
  bool ok6 = true;
  for (int b = 0; b < NBANK; b++) top->rd_addr[b] = 100;
  tick();
  edge();
  for (int b = 0; b < NBANK; b++) {
    int exp = (b == 7) ? 0xDEAD : 0;
    if ((int)top->rd_data[b] != exp) {
      ok6 = false;
      allPass = false;
      std::printf("  !! 阶段6 bank %2d = 0x%04x，期望 0x%04x\n", b,
                  (int)top->rd_data[b], exp);
    }
  }
  std::printf("阶段 6：读【本地地址 100】-> %s（只有 bank 7 写过）\n",
              ok6 ? "对" : "有错（见上）");

  // ---- 阶段 7：读从没写过的本地地址 0，期望全 0 ----
  bool ok7 = read_and_check(0, -1);
  if (!ok7) allPass = false;
  std::printf("阶段 7：读【本地地址 0】（没写过）-> %s\n",
              ok7 ? "全 0，对" : "有错（见上）");

  std::printf("\n%s\n", allPass ? "Pass" : "Failed");
  delete top;
  return allPass ? 0 : 1;
}
