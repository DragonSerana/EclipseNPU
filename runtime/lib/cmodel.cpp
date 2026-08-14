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

float CModel::readFP16(uint32_t addr) {
  return fp16ToFp32(loadU16(sram(addr)));
}

void CModel::writeFP16(uint32_t addr, float v) {
  storeU16(sram(addr), fp32ToFp16(v));
}

void CModel::exec(const Instruction &inst) {
  switch (inst.opcode) {
  case OpCode::DMA_LOAD: {
    const auto *desc = reinterpret_cast<const DMAParam *>(ddr(inst.descPtr));
    for (uint32_t i = 0; i < desc->rows; i++) {
      uint8_t *src = ddr(desc->ddrAddr + (desc->srcStride * i));
      // 在packet情况下，desc->dstStride与desc->cols * DTYPE_SIZE相等
      uint8_t *dst = sram(desc->sramAddr + (desc->dstStride * i));
      memcpy(dst, src, desc->cols * DTYPE_SIZE);
    }
    break;
  }
  case OpCode::DMA_STORE: {
    const auto *desc = reinterpret_cast<const DMAParam *>(ddr(inst.descPtr));
    for (uint32_t i = 0; i < desc->rows; i++) {
      uint8_t *src = sram(desc->sramAddr + (desc->srcStride * i));
      uint8_t *dst = ddr(desc->ddrAddr + (desc->dstStride * i));
      memcpy(dst, src, desc->cols * DTYPE_SIZE);
    }
    break;
  }
  case OpCode::MATMUL: {
    const auto *desc = reinterpret_cast<const MatmulParam *>(ddr(inst.descPtr));
    for (uint32_t m = 0; m < desc->M; m++) {
      for (uint32_t n = 0; n < desc->N; n++) {
        float acc = 0.0f;
        for (uint32_t k = 0; k < desc->K; k++) {
          acc += readFP16(desc->lhsAddr + ((m * desc->K) + k) * DTYPE_SIZE) *
                 readFP16(desc->rhsAddr + ((k * desc->N) + n) * DTYPE_SIZE);
        }
        if (desc->accumulate) {
          acc += readFP16(desc->dstAddr + ((m * desc->N) + n) * DTYPE_SIZE);
        }
        writeFP16(desc->dstAddr + ((m * desc->N) + n) * DTYPE_SIZE, acc);
      }
    }
    break;
  }
  case OpCode::ELEMENTWISE_ADD: {
    const auto *desc =
        reinterpret_cast<const EwiseAddParam *>(ddr(inst.descPtr));
    for (uint32_t i = 0; i < desc->n; i++) {
      const float lhs = readFP16(desc->lhsAddr + i * DTYPE_SIZE);
      const float rhs = readFP16(desc->rhsAddr + i * DTYPE_SIZE);
      writeFP16(desc->dstAddr + i * DTYPE_SIZE, lhs + rhs);
    }
    break;
  }
  case OpCode::ACT: {
    const auto *desc = reinterpret_cast<const ActParam *>(ddr(inst.descPtr));
    if (desc->kind != ActKind::RELU)
      ECLIPSE_ASSERT(false, "unsupported act kind");
    for (uint32_t i = 0; i < desc->n; i++) {
      const float src = readFP16(desc->srcAddr + i * DTYPE_SIZE);
      writeFP16(desc->dstAddr + i * DTYPE_SIZE, src > 0.0f ? src : 0.0f);
    }
    break;
  }
  case OpCode::SYNC: {
    // cmodel 顺序执行，指令天然串行完成，SYNC 无需额外动作
    break;
  }
  default:
    ECLIPSE_ASSERT(false, "Instruction must have opcode!");
  }
}
