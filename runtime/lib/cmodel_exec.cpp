// 指令执行：每条 opcode 一个函数，CModel::exec 只做派发。
#include "cmodel.h"
#include "eclipse_assert.h"
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

using namespace eclipse_runtime;

namespace {

/// REDUCE_ARGMAX 归约时跟着值一起走的 (值, 索引)
struct ValueIndex {
  float value;
  uint32_t index;
};

/// 最大值：NaN 传播；±0 取 +0（有 +0 就是 +0，全 -0 才是 -0）
float reduceMax(float a, float b) {
  if (std::isnan(a) || std::isnan(b))
    return std::numeric_limits<float>::quiet_NaN();
  if (a > b)
    return a;
  if (b > a)
    return b;
  if (a == 0.0f && b == 0.0f)
    return std::signbit(a) ? b : a;
  return a;
}

/// 值的规则与 reduceMax 一致；索引取第一个数值上等于值的元素
/// （值为 NaN 时取第一个 NaN），保证 MAX 和 ARGMAX 的结果自洽
ValueIndex reduceArgmax(const ValueIndex &a, const ValueIndex &b) {
  const float value = reduceMax(a.value, b.value);
  if (std::isnan(value))
    return {value, std::isnan(a.value) ? a.index : b.index};
  return {value, (a.value == value) ? a.index : b.index};
}

void execDmaLoad(CModel &sim, uint32_t descPtr) {
  const auto *desc = reinterpret_cast<const DMAParam *>(sim.ddr(descPtr));
  ECLIPSE_ASSERT((desc->sramAddr % 16 == 0) && (desc->ddrAddr % 16 == 0),
                 "dma load: address must be 16-byte aligned");
  for (uint32_t i = 0; i < desc->rows; i++) {
    const uint8_t *src = sim.ddr(desc->ddrAddr + (desc->srcStride * i));
    // 在packet情况下，desc->dstStride与desc->cols * DTYPE_SIZE相等
    uint8_t *dst = sim.sram(desc->sramAddr + (desc->dstStride * i));
    memcpy(dst, src, desc->cols * DTYPE_SIZE);
  }
}

void execDmaStore(CModel &sim, uint32_t descPtr) {
  const auto *desc = reinterpret_cast<const DMAParam *>(sim.ddr(descPtr));
  ECLIPSE_ASSERT((desc->sramAddr % 16 == 0) && (desc->ddrAddr % 16 == 0),
                 "dma store: address must be 16-byte aligned");
  for (uint32_t i = 0; i < desc->rows; i++) {
    const uint8_t *src = sim.sram(desc->sramAddr + (desc->srcStride * i));
    uint8_t *dst = sim.ddr(desc->ddrAddr + (desc->dstStride * i));
    memcpy(dst, src, desc->cols * DTYPE_SIZE);
  }
}

void execMatmul(CModel &sim, uint32_t descPtr) {
  const auto *desc = reinterpret_cast<const MatmulParam *>(sim.ddr(descPtr));
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
      for (uint32_t k = 0; k < desc->K; k++) {
        const uint32_t lhsOff =
            desc->transA ? (k * desc->M + m) : (m * desc->K + k);
        const uint32_t rhsOff =
            desc->transB ? (n * desc->K + k) : (k * desc->N + n);
        acc += sim.readFP16(desc->lhsAddr + lhsOff * DTYPE_SIZE) *
               sim.readFP16(desc->rhsAddr + rhsOff * DTYPE_SIZE);
      }
      if (desc->accumulate)
        acc += sim.readFP16(desc->dstAddr + ((m * desc->N) + n) * DTYPE_SIZE);

      sim.writeFP16(desc->dstAddr + ((m * desc->N) + n) * DTYPE_SIZE, acc);
    }
  }
}

void execElementwise(CModel &sim, uint32_t descPtr, OpCode op) {
  const auto *desc = reinterpret_cast<const EwiseParam *>(sim.ddr(descPtr));
  ECLIPSE_ASSERT(desc->cols != 0 && desc->rhsBlk != 0,
                 "ewize: cols/rhsBlk must be non-zero");
  ECLIPSE_ASSERT(desc->cols % desc->rhsBlk == 0,
                 "ewize: cols must be divisible by rhsBlk");

  const uint32_t n = desc->rows * desc->cols;
  for (uint32_t i = 0; i < n; i++) {
    const float lhs = sim.readFP16(desc->lhsAddr + i * DTYPE_SIZE);

    const uint32_t s = (i / desc->cols) * desc->rhsStride + i % desc->rhsBlk;
    const float rhs = sim.readFP16(desc->rhsAddr + s * DTYPE_SIZE);

    if (op == OpCode::ELEMENTWISE_ADD)
      sim.writeFP16(desc->dstAddr + i * DTYPE_SIZE, lhs + rhs);
    else if (op == OpCode::ELEMENTWISE_SUB)
      sim.writeFP16(desc->dstAddr + i * DTYPE_SIZE, lhs - rhs);
    else if (op == OpCode::ELEMENTWISE_MUL)
      sim.writeFP16(desc->dstAddr + i * DTYPE_SIZE, lhs * rhs);
    else if (op == OpCode::ELEMENTWISE_DIV) {
      ECLIPSE_ASSERT((rhs != 0), "The divisor cannot be zero.");
      sim.writeFP16(desc->dstAddr + i * DTYPE_SIZE, lhs / rhs);
    }
  }
}

void execAct(CModel &sim, uint32_t descPtr) {
  const auto *desc = reinterpret_cast<const ActParam *>(sim.ddr(descPtr));
  switch (desc->kind) {
  case ActKind::RELU: {
    for (uint32_t i = 0; i < desc->n; i++) {
      const float src = sim.readFP16(desc->srcAddr + i * DTYPE_SIZE);
      sim.writeFP16(desc->dstAddr + i * DTYPE_SIZE, src > 0.0f ? src : 0.0f);
    }
    break;
  }
  case ActKind::EXP: {
    for (uint32_t i = 0; i < desc->n; i++) {
      const float src = sim.readFP16(desc->srcAddr + i * DTYPE_SIZE);
      sim.writeFP16(desc->dstAddr + i * DTYPE_SIZE, std::exp(src));
    }
    break;
  }
  case ActKind::RSQRT: {
    for (uint32_t i = 0; i < desc->n; i++) {
      const float src = sim.readFP16(desc->srcAddr + i * DTYPE_SIZE);
      sim.writeFP16(desc->dstAddr + i * DTYPE_SIZE, 1.0 / std::sqrt(src));
    }
    break;
  }
  case ActKind::SILU: {
    for (uint32_t i = 0; i < desc->n; i++) {
      const float src = sim.readFP16(desc->srcAddr + i * DTYPE_SIZE);
      sim.writeFP16(desc->dstAddr + i * DTYPE_SIZE,
                    src / (1.0 + std::exp(-src)));
    }
    break;
  }
  default: {
    ECLIPSE_ASSERT(false, "unsupported act kind");
  }
  }
}

void execReduce(CModel &sim, uint32_t descPtr) {
  const auto *desc = reinterpret_cast<const ReduceParam *>(sim.ddr(descPtr));
  ECLIPSE_ASSERT(desc->rows != 0 && desc->cols != 0,
                 "reduce: rows/cols must be non-zero");
  ECLIPSE_ASSERT((desc->kind == ReduceKind::ARGMAX) == (desc->idxAddr != 0),
                 "reduce: idxAddr is only used by ARGMAX");
  switch (desc->kind) {
  case ReduceKind::MAX: {
    for (uint32_t i = 0; i < desc->rows; i++) {
      const uint32_t row = desc->srcAddr + i * desc->cols * DTYPE_SIZE;
      float max = sim.readFP16(row);
      for (uint32_t j = 1; j < desc->cols; j++)
        max = reduceMax(max, sim.readFP16(row + j * DTYPE_SIZE));
      sim.writeFP16(desc->dstAddr + i * DTYPE_SIZE, max);
    }
    break;
  }
  case ReduceKind::SUM:
  case ReduceKind::SQUARE_SUM: {
    // fp32 累加，只在写回时舍入一次；SQUARE_SUM 把平方折进归约输入端
    const bool square = desc->kind == ReduceKind::SQUARE_SUM;
    const auto scale = [square](float x) { return square ? x * x : x; };
    for (uint32_t i = 0; i < desc->rows; i++) {
      const uint32_t row = desc->srcAddr + i * desc->cols * DTYPE_SIZE;
      float sum = scale(sim.readFP16(row));
      for (uint32_t j = 1; j < desc->cols; j++)
        sum += scale(sim.readFP16(row + j * DTYPE_SIZE));
      sim.writeFP16(desc->dstAddr + i * DTYPE_SIZE, sum);
    }
    break;
  }
  case ReduceKind::ARGMAX: {
    for (uint32_t i = 0; i < desc->rows; i++) {
      const uint32_t row = desc->srcAddr + i * desc->cols * DTYPE_SIZE;
      ValueIndex result{sim.readFP16(row), 0};
      for (uint32_t j = 1; j < desc->cols; j++)
        result = reduceArgmax(
            result, ValueIndex{sim.readFP16(row + j * DTYPE_SIZE), j});
      sim.writeFP16(desc->dstAddr + i * DTYPE_SIZE, result.value);
      storeU32(sim.sram(desc->idxAddr + i * sizeof(uint32_t)), result.index);
    }
    break;
  }
  default: {
    ECLIPSE_ASSERT(false, "unsupported reduce kind");
  }
  }
}

} // namespace

void CModel::exec(const Instruction &inst) {
  switch (inst.opcode) {
  case OpCode::DMA_LOAD:
    execDmaLoad(*this, inst.descPtr);
    break;
  case OpCode::DMA_STORE:
    execDmaStore(*this, inst.descPtr);
    break;
  case OpCode::MATMUL:
    execMatmul(*this, inst.descPtr);
    break;
  case OpCode::ELEMENTWISE_ADD:
  case OpCode::ELEMENTWISE_SUB:
  case OpCode::ELEMENTWISE_MUL:
  case OpCode::ELEMENTWISE_DIV:
    execElementwise(*this, inst.descPtr, inst.opcode);
    break;
  case OpCode::ACT:
    execAct(*this, inst.descPtr);
    break;
  case OpCode::REDUCE:
    execReduce(*this, inst.descPtr);
    break;
  case OpCode::SYNC:
    // cmodel 顺序执行，指令天然串行完成，SYNC 无需额外动作
    break;
  default:
    ECLIPSE_ASSERT(false, "Instruction must have opcode!");
  }
}
