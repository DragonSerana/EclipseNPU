// 最小的 Verilator C++ testbench：驱动时钟、跑 40 拍、dump 波形。
// Verilator 的 TB 是 C++，没有 Verilog 的 initial 块。
#include "Vcounter.h"
#include "verilated.h"
#include "verilated_vcd_c.h"
#include <cstdio>

int main(int argc, char **argv) {
  Verilated::commandArgs(argc, argv);
  Verilated::traceEverOn(true);

  auto *top = new Vcounter;
  auto *vcd = new VerilatedVcdC;
  top->trace(vcd, 99);
  vcd->open("wave1.vcd");

  top->rst_n = 0;
  top->en = 1;
  // 注意 dump 的时间必须严格递增：每拍用 t*2 / t*2+1 两个时刻，
  // 写成 t*5 / t*5+5 会和下一拍的 t*5 撞上，Verilator 只警告不给样本。
  for (int t = 0; t < 40; t++) {
    top->clk = 0;
    top->eval();
    vcd->dump(t * 2);

    top->clk = 1;
    top->eval();
    vcd->dump(t * 2 + 1);

    if (t == 2) top->rst_n = 1;
    std::printf("cycle %2d  cnt=%3d\n", t, (int)top->cnt);
  }

  vcd->close();
  delete top;
  delete vcd;
  return 0;
}
