#ifndef ECLIPSE_ISA_H
#define ECLIPSE_ISA_H

#include <cstdint>
#include <stddef.h>

namespace eclipse {

// 内存映射与数据宽度（ISA 合同，与 docs/spec/isa-v0.1.md 一致）
constexpr uint32_t SRAM_ADDR = 0x10000000;
constexpr uint32_t DDR_ADDR = 0x80000000;

constexpr uint32_t SRAM_SIZE = 0x80000;
constexpr uint32_t DDR_SIZE = 0x40000000;

constexpr size_t DTYPE_SIZE = 2;

enum class OpCode : uint32_t {
  DMA_LOAD,
  DMA_STORE,
  MATMUL,
  ELEMENTWISE_ADD,
  ACT,
  SYNC
};

struct Instruction {
  OpCode opcode;
  uint32_t descPtr;
};

struct DMAOpcodeParam {
  uint32_t sramAddr;
  uint32_t ddrAddr;
  uint32_t rows;
  uint32_t cols;
  uint32_t srcStride;
  uint32_t dstStride;
};

struct MATMULOpcodeParam {
  uint32_t dstAddr;
  uint32_t rhsAddr;
  uint32_t lhsAddr;
  uint32_t M;
  uint32_t K;
  uint32_t N;
  uint32_t accumulate;
};

struct ElementwiseAddOpcodeParam {
  uint32_t dstAddr;
  uint32_t rhsAddr;
  uint32_t lhsAddr;
  uint32_t n;
};

struct ActOpcodeParam {
  uint32_t dstAddr;
  uint32_t srcAddr;
  uint32_t n;
  uint32_t kind;
  union {
    uint32_t extra[4];
  };
};

} // namespace eclipse

#endif
