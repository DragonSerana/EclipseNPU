#ifndef ECLIPSE_ISA_H
#define ECLIPSE_ISA_H

#include <cstdint>
#include <stddef.h>

namespace eclipse {

// Memory Define
constexpr uint32_t SRAM_ADDR = 0x10000000;
constexpr uint32_t DDR_ADDR = 0x80000000;

constexpr uint32_t SRAM_SIZE = 0x80000;
constexpr uint32_t DDR_SIZE = 0x40000000;

constexpr size_t DTYPE_SIZE = 2;

// Cycle Define
constexpr uint32_t DMA_BYTES_PER_CYCLE = 32;
constexpr uint32_t DMA_FIXED_OVERHEAD = 16;
constexpr uint32_t MAC_PER_CYCLE = 256;
constexpr uint32_t ELEM_PER_CYCLE = 128;

enum class OpCode : uint32_t {
  DMA_LOAD,
  DMA_STORE,
  MATMUL,
  ELEMENTWISE_ADD,
  ACT,
  SYNC
};

enum class ActKind : uint32_t { RELU };

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
};

struct EwiseAddParam {
  uint32_t dstAddr;
  uint32_t rhsAddr;
  uint32_t lhsAddr;
  uint32_t n; // 元素数
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

} // namespace eclipse

#endif
