#include "simulator.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace eclipse;

static constexpr uint32_t CMD_BASE = DDR_ADDR + 0x100;
static constexpr uint32_t DESC_STRIDE = 0x40;

static uint32_t nextDescAddr() {
  static uint32_t slot = 0;
  return CMD_BASE + slot++ * DESC_STRIDE;
}

static constexpr uint32_t A_SRAM = SRAM_ADDR + 0x0;
static constexpr uint32_t B_SRAM = SRAM_ADDR + 128 * 128 * DTYPE_SIZE;
static constexpr uint32_t C_SRAM = SRAM_ADDR + 128 * 128 * DTYPE_SIZE * 2;

static constexpr uint32_t A_DDR = DDR_ADDR + 0x10000;
static constexpr uint32_t B_DDR = DDR_ADDR + 0x20000;
static constexpr uint32_t C_DDR = DDR_ADDR + 0x30000;

static void pushDmaLoad(Simulator &sim, uint32_t sramAddr, uint32_t ddrAddr,
                        uint32_t cow, uint32_t col, uint32_t srcStride,
                        uint32_t dstStride) {
  DMAParam desc;
  desc.sramAddr = sramAddr;
  desc.ddrAddr = ddrAddr;
  desc.rows = cow;
  desc.cols = col;
  desc.srcStride = srcStride;
  desc.dstStride = dstStride;
  const uint32_t descAddr = nextDescAddr();
  sim.writeDDR(descAddr, &desc, sizeof(desc));
  sim.push(Instruction{OpCode::DMA_LOAD, descAddr});
}

// 不切分的版本
static void pushMatmulBlock(Simulator &sim, uint32_t accumulate) {
  MatmulParam desc;
  desc.dstAddr = C_SRAM;
  desc.lhsAddr = A_SRAM;
  desc.rhsAddr = B_SRAM;
  desc.accumulate = accumulate;
  desc.K = 128;
  desc.M = 128;
  desc.N = 128;
  const uint32_t descAddr = nextDescAddr();
  sim.writeDDR(descAddr, &desc, sizeof(desc));
  sim.push(Instruction{OpCode::MATMUL, descAddr});
}

static void pushMatmulBlockTile(Simulator &sim, uint32_t accumulate,
                                uint32_t K) {
  MatmulParam desc;
  desc.dstAddr = C_SRAM;
  desc.lhsAddr = A_SRAM;
  desc.rhsAddr = B_SRAM;
  desc.accumulate = accumulate;
  desc.K = K;
  desc.M = 128;
  desc.N = 128;
  const uint32_t descAddr = nextDescAddr();
  sim.writeDDR(descAddr, &desc, sizeof(desc));
  sim.push(Instruction{OpCode::MATMUL, descAddr});
}

static void pushDmaStore(Simulator &sim, uint32_t sramAddr, uint32_t ddrAddr) {
  DMAParam desc;
  desc.sramAddr = sramAddr;
  desc.ddrAddr = ddrAddr;
  desc.rows = 128;
  desc.cols = 128;
  // 这里因为DDR和SRAM都是packed,所以srcStride/dstStride都一样
  desc.srcStride = 128 * DTYPE_SIZE;
  desc.dstStride = desc.srcStride;
  const uint32_t descAddr = nextDescAddr();
  sim.writeDDR(descAddr, &desc, sizeof(desc));
  sim.push(Instruction{OpCode::DMA_STORE, descAddr});
}

static std::vector<uint8_t> readFile(const char *path) {
  FILE *fp = std::fopen(path, "rb");
  std::vector<uint8_t> data;
  if (!fp) {
    std::fprintf(stderr, "failed to open %s\n", path);
    std::exit(1);
  }
  std::fseek(fp, 0, SEEK_END);
  const long size = std::ftell(fp);
  std::fseek(fp, 0, SEEK_SET);
  data.resize(size);
  std::fread(data.data(), 1, size, fp);
  std::fclose(fp);
  return data;
}

static const char *opcodeName(OpCode opcode) {
  switch (opcode) {
  case OpCode::DMA_LOAD:
    return "DMA_LOAD";
  case OpCode::DMA_STORE:
    return "DMA_STORE";
  case OpCode::MATMUL:
    return "MATMUL";
  case OpCode::ELEMENTWISE_ADD:
    return "ELEMENTWISE_ADD";
  case OpCode::ACT:
    return "ACT";
  case OpCode::SYNC:
    return "SYNC";
  }
  return "UNKNOWN";
}

static void dumpTrace(const Simulator &sim, const char *path) {
  FILE *fp = std::fopen(path, "w");
  if (!fp) {
    std::fprintf(stderr, "failed to open trace file %s\n", path);
    std::exit(1);
  }

  for (const auto &e : sim.cycleLog()) {
    const Instruction &inst = e.inst;
    switch (inst.opcode) {
    case OpCode::DMA_LOAD:
    case OpCode::DMA_STORE: {
      const auto *desc =
          reinterpret_cast<const DMAParam *>(sim.cmodel().ddr(inst.descPtr));
      std::fprintf(fp,
                   "%-15s desc=0x%08x sram=0x%08x ddr=0x%08x rows=%u cols=%u "
                   "srcStride=%u dstStride=%u\n",
                   opcodeName(inst.opcode), inst.descPtr, desc->sramAddr,
                   desc->ddrAddr, desc->rows, desc->cols, desc->srcStride,
                   desc->dstStride);
      break;
    }
    case OpCode::MATMUL: {
      const auto *desc =
          reinterpret_cast<const MatmulParam *>(sim.cmodel().ddr(inst.descPtr));
      std::fprintf(
          fp,
          "MATMUL         desc=0x%08x dst=0x%08x lhs=0x%08x rhs=0x%08x "
          "M=%u N=%u K=%u acc=%u\n",
          inst.descPtr, desc->dstAddr, desc->lhsAddr, desc->rhsAddr, desc->M,
          desc->N, desc->K, desc->accumulate);
      break;
    }
    case OpCode::ELEMENTWISE_ADD: {
      const auto *desc = reinterpret_cast<const EwiseAddParam *>(
          sim.cmodel().ddr(inst.descPtr));
      std::fprintf(
          fp,
          "ELEMENTWISE_ADD desc=0x%08x dst=0x%08x lhs=0x%08x rhs=0x%08x "
          "n=%u\n",
          inst.descPtr, desc->dstAddr, desc->lhsAddr, desc->rhsAddr, desc->n);
      break;
    }
    case OpCode::ACT: {
      const auto *desc =
          reinterpret_cast<const ActParam *>(sim.cmodel().ddr(inst.descPtr));
      std::fprintf(fp,
                   "ACT            desc=0x%08x dst=0x%08x src=0x%08x n=%u "
                   "kind=%u\n",
                   inst.descPtr, desc->dstAddr, desc->srcAddr, desc->n,
                   static_cast<unsigned>(desc->kind));
      break;
    }
    case OpCode::SYNC:
      std::fprintf(fp, "SYNC\n");
      break;
    }
  }

  std::fclose(fp);
}

int main(int argc, char **argv) {
  if (argc < 4) {
    std::fprintf(stderr, "usage: %s <A.raw> <B.raw> <C.raw> [trace.easm]\n",
                 argv[0]);
    return 1;
  }
  const char *tracePath = (argc >= 5) ? argv[4] : "golden.easm";

  const auto aBytes = readFile(argv[1]);
  const auto bBytes = readFile(argv[2]);

  Simulator sim;
  sim.writeDDR(A_DDR, aBytes.data(), aBytes.size());
  sim.writeDDR(B_DDR, bBytes.data(), bBytes.size());
  int tile = 8;
  for (int i = 0; i < tile; i++) {
    pushDmaLoad(sim, A_SRAM, A_DDR + i * 128 / tile * DTYPE_SIZE, 128,
                128 / tile, 128 * DTYPE_SIZE, 128 / tile * DTYPE_SIZE);
    pushDmaLoad(sim, B_SRAM, B_DDR + i * 128 / tile * 128 * DTYPE_SIZE,
                128 / tile, 128, 128 * DTYPE_SIZE, 128 * DTYPE_SIZE);
    sim.push(Instruction{OpCode::SYNC, 0});
    pushMatmulBlockTile(sim, (i == 0) ? 0 : 1, 128 / tile);
    sim.push(Instruction{OpCode::SYNC, 0});
  }
  sim.push(Instruction{OpCode::SYNC, 0});
  pushDmaStore(sim, C_SRAM, C_DDR);

  sim.run();
  dumpTrace(sim, tracePath);

  FILE *fp = std::fopen(argv[3], "wb");
  std::fwrite(sim.cmodel().sram(C_SRAM), 1, 128 * 128 * DTYPE_SIZE, fp);
  std::fclose(fp);

  for (int i = 0; i < 4; i++)
    std::printf("C[0][%d] = %f\n", i,
                sim.cmodel().readFP16(C_SRAM + i * DTYPE_SIZE));
  for (const auto &e : sim.cycleLog())
    std::printf("op=%d cycles=%llu\n", static_cast<int>(e.inst.opcode),
                static_cast<unsigned long long>(e.cycles));
  std::printf("total = %llu\n",
              static_cast<unsigned long long>(sim.totalCycles()));
  return 0;
}