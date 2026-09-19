// 周期模型：每条 opcode 一个函数，CModel::computeCycles 只做派发。
#include "cmodel.h"
#include "eclipse_assert.h"
#include <cstdint>

using namespace eclipse_runtime;

namespace {

uint64_t cyclesDma(const CModel &sim, uint32_t descPtr, bool isLoad) {
  const auto *desc = reinterpret_cast<const DMAParam *>(sim.ddr(descPtr));
  const uint32_t stride = isLoad ? desc->srcStride : desc->dstStride;
  const uint32_t rowBytes = desc->cols * DTYPE_SIZE;
  uint64_t bursts = 0;
  for (uint32_t i = 0; i < desc->rows; i++) {
    const uint32_t start = desc->ddrAddr + stride * i;
    // 结尾所在块 - 开头所在块
    bursts +=
        (start + rowBytes - 1) / DMA_BURST_BYTES - start / DMA_BURST_BYTES + 1;
  }
  return ceilDiv(bursts * DMA_BURST_BYTES, DMA_BYTES_PER_CYCLE) +
         DMA_FIXED_OVERHEAD;
}

uint64_t cyclesMatmul(const CModel &sim, uint32_t descPtr) {
  const auto *desc = reinterpret_cast<const MatmulParam *>(sim.ddr(descPtr));
  return ceilDiv(desc->M * desc->N, MAC_PER_CYCLE) * desc->K;
}

/// 四则走 SIMD 引擎
uint64_t cyclesElementwise(const CModel &sim, uint32_t descPtr) {
  const auto *desc = reinterpret_cast<const EwiseParam *>(sim.ddr(descPtr));
  return ceilDiv(static_cast<uint64_t>(desc->rows) * desc->cols,
                 ELEM_PER_CYCLE);
}

uint64_t cyclesAct(const CModel &sim, uint32_t descPtr) {
  // RELU 走 SIMD 满速；EXP/RSQRT/SILU 走 SFU，约 1/4 吞吐 + 一次固定开销。
  const auto *desc = reinterpret_cast<const ActParam *>(sim.ddr(descPtr));
  if (desc->kind == ActKind::RELU)
    return ceilDiv(desc->n, ELEM_PER_CYCLE);
  return ceilDiv(desc->n, SFU_ELEM_PER_CYCLE) + ACT_FIXED_OVERHEAD;
}

uint64_t cyclesReduce(const CModel &sim, uint32_t descPtr) {
  // 每行独立：ceil(cols / ELEM_PER_CYCLE) 拍读数 + 跨 lane 归约树
  const auto *desc = reinterpret_cast<const ReduceParam *>(sim.ddr(descPtr));
  const uint32_t elemPerCycle = (desc->kind == ReduceKind::ARGMAX)
                                    ? ARGMAX_ELEM_PER_CYCLE
                                    : ELEM_PER_CYCLE;
  return ceilDiv(static_cast<uint64_t>(desc->rows) * desc->cols, elemPerCycle) +
         static_cast<uint64_t>(desc->rows) * REDUCE_TREE_STEPS;
}

} // namespace

uint64_t CModel::computeCycles(const Instruction &inst) const {
  switch (inst.opcode) {
  case OpCode::DMA_LOAD:
    return cyclesDma(*this, inst.descPtr, true);
  case OpCode::DMA_STORE:
    return cyclesDma(*this, inst.descPtr, false);
  case OpCode::MATMUL:
    return cyclesMatmul(*this, inst.descPtr);
  case OpCode::ELEMENTWISE_ADD:
  case OpCode::ELEMENTWISE_SUB:
  case OpCode::ELEMENTWISE_MUL:
  case OpCode::ELEMENTWISE_DIV:
    return cyclesElementwise(*this, inst.descPtr);
  case OpCode::ACT:
    return cyclesAct(*this, inst.descPtr);
  case OpCode::REDUCE:
    return cyclesReduce(*this, inst.descPtr);
  case OpCode::SYNC:
    // 目前为串行，后续排流水后根据木桶效应取值
    return 0;
  default:
    ECLIPSE_ASSERT(false, "computeCycles: opcode not implemented");
    return 0;
  }
}
