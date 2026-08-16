#include "cmodel.h"
#include "eclipse_assert.h"
#include <cmath>
#include <cstdint>
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

const uint8_t *CModel::ddr(uint32_t addr) const {
  ECLIPSE_ASSERT((addr >= DDR_ADDR) && (addr < (DDR_ADDR + DDR_SIZE)),
                 "ddr addr out of range");
  return ddr_.data() + (addr - DDR_ADDR);
}

uint8_t *CModel::sram(uint32_t addr) {
  ECLIPSE_ASSERT((addr >= SRAM_ADDR) && (addr < (SRAM_ADDR + SRAM_SIZE)),
                 "sram addr out of range");
  return sram_.data() + (addr - SRAM_ADDR);
}

const uint8_t *CModel::sram(uint32_t addr) const {
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
    ECLIPSE_ASSERT((desc->sramAddr % 16 == 0) && (desc->ddrAddr % 16 == 0),
                   "dma load: address must be 16-byte aligned");
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
    ECLIPSE_ASSERT((desc->sramAddr % 16 == 0) && (desc->ddrAddr % 16 == 0),
                   "dma store: address must be 16-byte aligned");
    for (uint32_t i = 0; i < desc->rows; i++) {
      uint8_t *src = sram(desc->sramAddr + (desc->srcStride * i));
      uint8_t *dst = ddr(desc->ddrAddr + (desc->dstStride * i));
      memcpy(dst, src, desc->cols * DTYPE_SIZE);
    }
    break;
  }
  case OpCode::MATMUL: {
    const auto *desc = reinterpret_cast<const MatmulParam *>(ddr(inst.descPtr));
    const uint32_t dstEnd = desc->dstAddr + desc->M * desc->N * DTYPE_SIZE;
    const uint32_t lhsEnd = desc->lhsAddr + desc->M * desc->K * DTYPE_SIZE;
    const uint32_t rhsEnd = desc->rhsAddr + desc->K * desc->N * DTYPE_SIZE;
    ECLIPSE_ASSERT((desc->dstAddr >= lhsEnd || desc->lhsAddr >= dstEnd) &&
                       (desc->dstAddr >= rhsEnd || desc->rhsAddr >= dstEnd) &&
                       (desc->lhsAddr >= rhsEnd || desc->rhsAddr >= lhsEnd),
                   "matmul: dst/lhs/rhs must not overlap");
    for (uint32_t m = 0; m < desc->M; m++) {
      for (uint32_t n = 0; n < desc->N; n++) {
        float acc = 0.0f;
        for (uint32_t k = 0; k < desc->K; k++)
          acc += readFP16(desc->lhsAddr + ((m * desc->K) + k) * DTYPE_SIZE) *
                 readFP16(desc->rhsAddr + ((k * desc->N) + n) * DTYPE_SIZE);
        if (desc->accumulate)
          acc += readFP16(desc->dstAddr + ((m * desc->N) + n) * DTYPE_SIZE);

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

uint64_t CModel::computeCycles(const Instruction &inst) const {
  uint64_t cycles = 0;
  switch (inst.opcode) {
  case OpCode::DMA_LOAD:
  case OpCode::DMA_STORE: {
    const auto *desc = reinterpret_cast<const DMAParam *>(ddr(inst.descPtr));
    const uint32_t stride =
        (inst.opcode == OpCode::DMA_LOAD) ? desc->srcStride : desc->dstStride;
    const uint32_t rowBytes = desc->cols * DTYPE_SIZE;
    uint64_t bursts = 0;
    for (uint32_t i = 0; i < desc->rows; i++) {
      const uint32_t start = desc->ddrAddr + stride * i;
      // 结尾所在块 - 开头所在块
      bursts += (start + rowBytes - 1) / DMA_BURST_BYTES -
                start / DMA_BURST_BYTES + 1;
    }
    cycles = ceil(((float)bursts * DMA_BURST_BYTES) / DMA_BYTES_PER_CYCLE) +
             DMA_FIXED_OVERHEAD;
    break;
  }
  case OpCode::MATMUL: {
    const auto *desc = reinterpret_cast<const MatmulParam *>(ddr(inst.descPtr));
    cycles = ceil((float)desc->M * desc->N / MAC_PER_CYCLE) * desc->K;
    break;
  }
  // 这两个指令使用SIMD引擎
  case OpCode::ELEMENTWISE_ADD:
  case OpCode::ACT: {
    uint32_t n;
    if (inst.opcode == OpCode::ELEMENTWISE_ADD)
      n = reinterpret_cast<const EwiseAddParam *>(ddr(inst.descPtr))->n;
    else
      n = reinterpret_cast<const ActParam *>(ddr(inst.descPtr))->n;
    cycles = ceil((float)n / ELEM_PER_CYCLE);
    break;
  }
  case OpCode::SYNC: {
    // 目前为串行，后续排流水后根据木桶效应取值
    cycles = 0;
    break;
  }
  default:
    ECLIPSE_ASSERT(false, "computeCycles: opcode not implemented");
  }

  return cycles;
}