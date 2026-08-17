#ifndef SIMULATOR_H
#define SIMULATOR_H

#include "cmodel.h"
#include "eclipse_isa.h"
#include <deque>
#include <vector>

namespace eclipse {

struct CycleEntry {
  Instruction inst;
  uint64_t cycles;
};

class Simulator {
private:
  CModel cmodel_;
  std::deque<Instruction> queue_;
  uint64_t totalCycles_ = 0;
  std::vector<CycleEntry> log_;

public:
  void writeDdr(uint32_t addr, const void *buffer, size_t size);
  void push(const Instruction &inst);
  void run();
  uint64_t totalCycles() const;
  const std::vector<CycleEntry> &cycleLog() const;
  CModel &cmodel();
  const CModel &cmodel() const;
};

} // namespace eclipse

#endif
