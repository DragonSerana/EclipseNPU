#include "cmodel.h"
#include "eclipse_assert.h"

using namespace eclipse;

CModel::CModel() {
  sram_.resize(SRAM_SIZE);
  ddr_.resize(DDR_SIZE);
}
	
uint8_t *CModel::ddr(uint32_t addr) {
  ECLIPSE_ASSERT((addr >= DDR_ADDR) && (addr < (DDR_ADDR + DDR_SIZE)),
                 "ddr addr out of range");
  return ddr_.data() + (addr - DDR_ADDR);
}

uint8_t *CModel::sram(uint32_t addr) {
  ECLIPSE_ASSERT((addr >= SRAM_ADDR) && (addr < (SRAM_ADDR + SRAM_SIZE)),
                 "sram addr out of range");
  return sram_.data() + (addr - SRAM_ADDR);
}

void CModel::exec(const Instruction &inst) {
  switch (inst.opcode) {
    case OpCode::DMA_LOAD: {
      
      
      break;
    }

    default: 
      ECLIPSE_ASSERT(false, "Instruction must have opcode!");
  }
}
