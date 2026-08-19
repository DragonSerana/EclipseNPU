#include "simulator.h"

#include <cstdint>
#include <cstdio>
#include <vector>

using namespace eclipse;

static constexpr uint32_t CMD_BASE = DDR_ADDR + 0x100;
static constexpr uint32_t DESC_STRIDE = 0x40;

static uint32_t nextDescAddr() {
  static uint32_t slot = 0;
  return CMD_BASE + slot++ * DESC_STRIDE;
}

static constexpr uint32_t A_SRAM = SRAM_ADDR + 0x0;
static constexpr uint32_t B_SRAM = SRAM_ADDR + 128*128*DTYPE_SIZE;
static constexpr uint32_t C_SRAM = SRAM_ADDR + 128*128*DTYPE_SIZE*2;

static constexpr uint32_t A_DDR = DDR_ADDR + 0x10000;
static constexpr uint32_t B_DDR = DDR_ADDR + 0x20000;
static constexpr uint32_t C_DDR = DDR_ADDR + 0x30000;

static void pushDmaLoad(Simulator &sim, uint32_t sramAddr, uint32_t ddrAddr, 
    uint32_t cow, uint32_t col, uint32_t srcStride, uint32_t dstStride) {
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

//不切分的版本
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

static void pushMatmulBlockTile(Simulator &sim, uint32_t accumulate, uint32_t K) {
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
  if (!fp)
    return data;
  std::fseek(fp, 0, SEEK_END);
  const long size = std::ftell(fp);
  std::fseek(fp, 0, SEEK_SET);
  data.resize(size);
  std::fread(data.data(), 1, size, fp);
  std::fclose(fp);
  return data;
}

int main(int argc, char **argv) {
  const auto aBytes = readFile(argv[1]);
  const auto bBytes = readFile(argv[2]);

  Simulator sim;
  sim.writeDDR(A_DDR, aBytes.data(), aBytes.size());
  sim.writeDDR(B_DDR, bBytes.data(), bBytes.size());
  int tile = 8;
  for(int i = 0; i < tile; i++) {
    pushDmaLoad(sim, A_SRAM, A_DDR+i*128/tile*DTYPE_SIZE, 128, 128/tile, 128*DTYPE_SIZE, 128/tile*DTYPE_SIZE);
    pushDmaLoad(sim, B_SRAM, B_DDR+i*128/tile*128*DTYPE_SIZE, 128/tile, 128, 128*DTYPE_SIZE, 128*DTYPE_SIZE);
    sim.push(Instruction{OpCode::SYNC, 0});
    pushMatmulBlockTile(sim, (i==0)?0:1, 128/tile);
  }
  sim.push(Instruction{OpCode::SYNC, 0});
  pushDmaStore(sim, C_SRAM, C_DDR);

  sim.run();

  FILE *fp = std::fopen(argv[3], "wb");
  std::fwrite(sim.cmodel().sram(C_SRAM), 1, 128 * 128 * DTYPE_SIZE, fp);
  std::fclose(fp);

  for (int i = 0; i < 4; i++)
    std::printf("C[0][%d] = %f\n", i, sim.cmodel().readFP16(C_SRAM + i * DTYPE_SIZE));
  for (const auto &e : sim.cycleLog())
    std::printf("op=%d cycles=%llu\n", static_cast<int>(e.inst.opcode),
                static_cast<unsigned long long>(e.cycles));
  std::printf("total = %llu\n", static_cast<unsigned long long>(sim.totalCycles()));
  return 0;
}