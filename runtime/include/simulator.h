#ifndef SIMULATOR_H
#define SIMULATOR_H

#include "cmodel.h"
#include "eclipse_isa.h"
#include <deque>
#include <vector>

namespace eclipse {

class Simulator {
private:
  CModel cmodel_;
  std::deque<Instruction> queue_; // 指令队列
  uint64_t total_cycles_ = 0;

public:
  bool write_ddr(uint32_t addr, const void *buffer, size_t size);
  bool push();      // 将指令压入指令队列
  bool run();       // 开始执行指令
  uint64_t cycle(); // 统计cycle数据
};

} // namespace eclipse

#endif