#include "Vpipe_slice.h"
#include "verilated.h"
#include "verilated_vcd_c.h"
#include <cstdio>

void tick(Vpipe_slice *top) {
    top->clk = 0;
    top->eval();
}

void edge(Vpipe_slice *top) {
    top->clk = 1;
    top->eval();
}

int main(int argc, char **argv) {
  Verilated::commandArgs(argc, argv);
  Verilated::traceEverOn(true);

  auto *top = new Vpipe_slice;
  auto *vcd = new VerilatedVcdC;
  top->trace(vcd, 99);
  vcd->open("wave1.vcd");

  top->rst_n = 0;
  top->s_valid = 1;
  top->m_ready = 1;

  bool allPass = true;

  for (int t = 0; t < 40; t++) {
    top->s_data = t;

    tick(top);
    vcd->dump(t * 2);

    if (top->s_valid && top->s_ready && top->m_valid) {
      // 因为这里的m_data,是获取上一cycle的数据，此次上升沿结束后，寄存器的值才会变成本次
      if ( top->rst_n && top->m_data != t-1 ) {
        std::printf("top->rst_n = %d, t-1 = %2d, m_data = %3d\n", top->rst_n, t-1, (int)top->m_data);
        allPass = false;
      }
    }

    edge(top);
    vcd->dump(t * 2 + 1);

    if (t == 2) top->rst_n = 1;
  }

  if( allPass ) {
    printf("Pass\n");
  } else {
    printf("Failed\n");
  }

  vcd->close();
  delete top;
  delete vcd;
  return 0;
}
