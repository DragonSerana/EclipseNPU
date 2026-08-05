#ifndef CMODEL_H
#define CMODEL_H

#include "eclipse_isa.h"
#include <vector>

namespace eclipse {

class CModel { // 对芯片的模拟
private:
  std::vector<uint8_t> ddr_, sram_;

public:
  void exec(const Instruction &inst);
  uint8_t *ddr(uint32_t addr);
  uint8_t *sram(uint32_t addr);
  uint64_t compute_cycles(const Instruction &inst);
};

} // namespace eclipse

#endif