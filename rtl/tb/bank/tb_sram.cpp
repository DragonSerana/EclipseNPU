// R2.1 激励：单 bank（1R1W）+ 同步读
//
// 对应 rtl/sram.sv 的当前版本：
//     读口  always_ff @(posedge clk) if (!rst_n) rdata <= 0; else rdata <= mem[raddr];
//     写口  always_ff @(posedge clk) if (we) mem[waddr] <= wdata;
//
// 这一份同时覆盖两件事：
//
// 【实验 1 读不破坏】地址 5 在第 2、3、5 拍被反复读出 0x1111，
//   而从第 0 拍之后就没再写过它 —— 读不会清掉数据。
//
// 【实验 2 同步读有 1 拍延迟】唯一的规律：
//
//     rdata(第 t 拍) = mem[ 第 (t-1) 拍 的 raddr ]
//
//   rdata 是寄存器，它在沿上采样 mem[raddr]，下一拍才拿得到。
//   看下面那张表里的「上拍raddr」列就明白了 —— rdata 永远跟着上一拍的地址走。
//
//   对照组合读版本（assign rdata = mem[raddr]）会得到：
//     t:      0      1      2      3      4      5
//     rdata:  0x0000 0x1111 0x1111 0x2222 0x1111 0x2222   ← 跟【本拍】地址走
//   同步读版本整体后移一拍。

#include "Vsram.h"
#include "verilated.h"
#include "verilated_vcd_c.h"
#include <cstdio>

static Vsram *top;
static VerilatedVcdC *vcd;

static void tick() { top->clk = 0; top->eval(); } // 这一拍开始
static void edge() { top->clk = 1; top->eval(); } // 上升沿：读写都在这时生效

int main(int argc, char **argv) {
  Verilated::commandArgs(argc, argv);
  Verilated::traceEverOn(true);

  top = new Vsram;
  vcd = new VerilatedVcdC;
  top->trace(vcd, 99);
  vcd->open("wave_sram.vcd");

  // ---- 复位 ----
  // rdata 现在是寄存器了，必须给初值；mem 不能复位，所以只能靠后面先写再读。
  top->rst_n = 0;
  top->we = 0;
  top->waddr = 0;
  top->wdata = 0;
  top->raddr = 0;
  tick();
  edge(); // 复位第 1 拍
  tick();
  edge(); // 复位第 2 拍
  top->rst_n = 1;

  struct Stim {
    int we, waddr, wdata, raddr;
    int expect; // rdata 的期望值
  };
  const Stim stim[6] = {
      {1, 5, 0x1111, 0, 0x0000}, // 上拍=复位   → rdata 被复位成 0
      {1, 6, 0x2222, 5, 0x0000}, // 上拍 raddr=0 → mem[0] 从没写过，是 0
      {0, 0, 0, 5, 0x1111},      // 上拍 raddr=5 → mem[5]
      {0, 0, 0, 6, 0x1111},      // 上拍 raddr=5 → 还是 mem[5]   ← 地址 5 第二次被读
      {0, 0, 0, 5, 0x2222},      // 上拍 raddr=6 → mem[6]
      {0, 0, 0, 6, 0x1111},      // 上拍 raddr=5 → mem[5]
  };

  bool allPass = true;
  std::printf(" t | we waddr wdata  | raddr | 上拍raddr | rdata  | 期望   | 判定\n");
  std::printf("---+-----------------+-------+-----------+--------+--------+--------\n");

  for (int t = 0; t < 6; t++) {
    top->we = stim[t].we;
    top->waddr = stim[t].waddr;
    top->wdata = stim[t].wdata;
    top->raddr = stim[t].raddr;

    tick(); // clk=0：rdata 已经是上一拍沿锁住的值，此刻读它才是稳的
    vcd->dump(t * 2);

    int got = (int)top->rdata;
    bool ok = (got == stim[t].expect);
    if (!ok) allPass = false;

    std::printf(
        " %d |  %d   %3d  0x%04x |  %3d  |    %3d    | 0x%04x | 0x%04x | %s\n",
        t, stim[t].we, stim[t].waddr, stim[t].wdata, stim[t].raddr,
        t == 0 ? -1 : stim[t - 1].raddr, got, stim[t].expect,
        ok ? "ok" : "!! 不符");

    edge(); // clk=1：写在这时生效，rdata 也在这时锁住 mem[raddr]
    vcd->dump(t * 2 + 1);
  }

  std::printf("\n%s\n", allPass ? "Pass" : "Failed");

  vcd->close();
  delete top;
  delete vcd;
  return allPass ? 0 : 1;
}
