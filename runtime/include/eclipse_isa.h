#ifndef ECLIPSE_ISA_H
#define ECLIPSE_ISA_H

#include <cstdint>
#include <stddef.h>

namespace eclipse_runtime {

// Memory Define
constexpr uint32_t SRAM_ADDR = 0x10000000;
constexpr uint32_t DDR_ADDR = 0x40000000;

constexpr uint32_t SRAM_SIZE = 0x80000;
constexpr uint32_t DDR_SIZE = 0x80000000; // v0.2: 1GB -> 2GB

static_assert(static_cast<uint64_t>(DDR_ADDR) + DDR_SIZE <= 0xFFFFFFFFull,
              "The DDR range upper bound overflows uint32, so the "
              "out-of-bounds check will wrap around.");

constexpr size_t DTYPE_SIZE = 2;

// Cycle Define
constexpr uint32_t DMA_BYTES_PER_CYCLE = 32;
constexpr uint32_t DMA_FIXED_OVERHEAD = 16;
constexpr uint32_t DMA_BURST_BYTES = 16;
constexpr uint32_t MAC_PER_CYCLE = 256;
constexpr uint32_t ELEM_PER_CYCLE = 128;
constexpr uint32_t SFU_ELEM_PER_CYCLE =
    ELEM_PER_CYCLE / 4; // Special Function Unit，超越函数使用
constexpr uint32_t ACT_FIXED_OVERHEAD = 8;

enum class OpCode : uint32_t {
  DMA_LOAD,
  DMA_STORE,
  MATMUL,
  ELEMENTWISE_ADD,
  ELEMENTWISE_SUB,
  ELEMENTWISE_MUL,
  ELEMENTWISE_DIV,
  ACT,
  SYNC
};

enum class ActKind : uint32_t { RELU, EXP, RSQRT, SILU };

struct Instruction {
  OpCode opcode;
  uint32_t descPtr;
};

struct DMAParam {
  uint32_t sramAddr;
  uint32_t ddrAddr;
  uint32_t rows;
  uint32_t cols;
  uint32_t srcStride;
  uint32_t dstStride;
};

struct MatmulParam {
  uint32_t dstAddr;
  uint32_t rhsAddr;
  uint32_t lhsAddr;
  uint32_t M;
  uint32_t K;
  uint32_t N;
  uint32_t accumulate;
  bool transA;
  bool transB;
};

struct EwiseParam {
  uint32_t dstAddr;
  uint32_t rhsAddr;
  uint32_t lhsAddr;
  uint32_t rows;      // 输出行数
  uint32_t cols;      // 输出行宽
  uint32_t rhsBlk;    // rhs 的行内重复周期
  uint32_t rhsStride; // rhs 每行前进多少元素（0 = 行广播）
};

struct ActParam {
  uint32_t dstAddr;
  uint32_t srcAddr;
  uint32_t n; // 元素数
  ActKind kind;
  union {
    uint32_t extra[4];
  };
};

} // namespace eclipse_runtime

#endif
