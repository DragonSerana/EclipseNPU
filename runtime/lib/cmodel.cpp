#include "cmodel.h"
#include "eclipse_assert.h"
#include <cstring>

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
      const auto* desc = reinterpret_cast<const dma_opcode_param *>(ddr(inst.desc_ptr));
      for(uint32_t i = 0; i < desc->rows; i++) {
        uint8_t *src = ddr(desc->ddr_addr + (desc->src_stride * i));
        // 在packet情况下，desc->dst_stride与desc->cols * DTYPE_SIZE相等
        uint8_t *dst = sram(desc->sram_addr + (desc->dst_stride * i)); 
        memcpy(dst, src, desc->cols * DTYPE_SIZE);
      }
      break;
    }
    case OpCode::DMA_STORE: {
      const auto* desc = reinterpret_cast<const dma_opcode_param *>(ddr(inst.desc_ptr));
      for(uint32_t i = 0; i < desc->rows; i++) {
        uint8_t *src = sram(desc->sram_addr + (desc->src_stride * i));
        uint8_t *dst = ddr(desc->ddr_addr + (desc->dst_stride * i)); 
        memcpy(dst, src, desc->cols * DTYPE_SIZE);
      }
      break;
    }

    default: 
      ECLIPSE_ASSERT(false, "Instruction must have opcode!");
  }
}
