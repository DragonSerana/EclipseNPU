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
  top->m_ready = 0;

  bool allPass = true;
  int send_cnt = 0;
  int recv_cnt = 0;

  for (int t = 0; t < 40; t++) {
    top->s_data = send_cnt;

    tick(top);
    vcd->dump(t * 2);

    if (top->s_valid && top->s_ready && top->rst_n)
      send_cnt++;    
    if (top->m_valid && top->m_ready) {
      printf("ly @@@ top->m_data = %d, recv_cnt = %d \n", top->m_data, recv_cnt);
      if (top->m_data != recv_cnt) allPass = false;
      recv_cnt++;
    }

    edge(top);
    vcd->dump(t * 2 + 1);

    if (t == 2) top->rst_n = 1;
    if (t % 4 == 0) {
      top->m_ready = 1;
    }
    else 
      top->m_ready = 0;
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
