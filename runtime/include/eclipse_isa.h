#ifndef ECLIPSE_ISA_H
#define ECLIPSE_ISA_H

#include <cstdint>
#include <stddef.h>

namespace eclipse {

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
  uint32_t desc_ptr;
};

struct dma_opcode_param {
  uint32_t sram_addr;
  uint32_t ddr_addr;
  uint32_t rows;
  uint32_t cols;
  uint32_t src_stride;
  uint32_t dst_stride;
};

struct matmul_opcode_param {
  uint32_t dst_addr;
  uint32_t rhs_addr;
  uint32_t lhs_addr;
  uint32_t M;
  uint32_t K;
  uint32_t N;
  uint32_t accumulate;
};

struct elementwise_add_opcode_param {
  uint32_t dst_addr;
  uint32_t rhs_addr;
  uint32_t lhs_addr;
  uint32_t n;
};

struct act_opcode_param {
  uint32_t dst_addr;
  uint32_t src_addr;
  uint32_t n;
  uint32_t kind;
  union {
    uint32_t extra[4];
  };
};

} // namespace eclipse

#endif