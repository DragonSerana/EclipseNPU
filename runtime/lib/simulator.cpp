#include "simulator.h"
#include <cstdint>
#include <cstring>

using namespace eclipse_runtime;

void Simulator::writeDDR(uint32_t addr, const void *buffer, size_t size) {
  memcpy(cmodel_.ddr(addr), buffer, size);
}

void Simulator::push(const Instruction &inst) { queue_.push_back(inst); }

void Simulator::run() {
  while (!queue_.empty()) {
    Instruction inst = queue_.front();
    queue_.pop_front();
    cmodel_.exec(inst);

    uint64_t cycles = cmodel_.computeCycles(inst);
    log_.push_back({inst, cycles});
    totalCycles_ += cycles;
  }
}

uint64_t Simulator::totalCycles() const { return totalCycles_; }

const std::vector<CycleEntry> &Simulator::cycleLog() const { return log_; }

CModel &Simulator::cmodel() { return cmodel_; }
const CModel &Simulator::cmodel() const { return cmodel_; }
