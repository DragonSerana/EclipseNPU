// eclipse-run: 读取 .easm，驱动 Simulator 对拍。
// 用法: eclipse-run <input.easm> <out.raw> [in.raw ...]

#include "simulator.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

using namespace eclipse_runtime;

namespace {

uint32_t hexVal(const std::string &s) {
  return static_cast<uint32_t>(std::strtoul(s.c_str(), nullptr, 0));
}

long intVal(const std::string &s) {
  return std::strtol(s.c_str(), nullptr, 0);
}

struct Insn {
  std::string op;
  std::map<std::string, std::string> fields;
};

std::vector<Insn> parseEasm(const char *path) {
  std::vector<Insn> insns;
  FILE *fp = std::fopen(path, "r");
  if (!fp) {
    std::fprintf(stderr, "eclipse-run: cannot open '%s'\n", path);
    std::exit(1);
  }
  char line[4096];
  while (std::fgets(line, sizeof(line), fp)) {
    std::string s(line);
    s.erase(s.find_last_not_of(" \t\r\n") + 1);
    if (s.empty())
      continue;
    Insn insn;
    size_t pos = 0;
    while (pos < s.size()) {
      while (pos < s.size() && (s[pos] == ' ' || s[pos] == '\t'))
        ++pos;
      size_t end = pos;
      while (end < s.size() && s[end] != ' ' && s[end] != '\t')
        ++end;
      std::string tok = s.substr(pos, end - pos);
      pos = end;
      if (tok.empty())
        continue;
      if (insn.op.empty()) {
        insn.op = tok;
      } else {
        size_t eq = tok.find('=');
        if (eq != std::string::npos)
          insn.fields[tok.substr(0, eq)] = tok.substr(eq + 1);
      }
    }
    insns.push_back(insn);
  }
  std::fclose(fp);
  return insns;
}

} // namespace

int main(int argc, char **argv) {
  if (argc < 4) {
    std::fprintf(stderr,
                 "usage: eclipse-run <input.easm> <out.raw> [in.raw ...]\n");
    return 2;
  }
  const char *easmPath = argv[1];
  const char *outPath = argv[2];

  std::vector<Insn> insns = parseEasm(easmPath);
  Simulator sim;

  // 把输入 tensor 写进 DDR ABI 地址 0x80010000, 0x80020000, ...
  for (int i = 3; i < argc; ++i) {
    FILE *fp = std::fopen(argv[i], "rb");
    if (!fp) {
      std::fprintf(stderr, "eclipse-run: cannot open '%s'\n", argv[i]);
      return 2;
    }
    std::fseek(fp, 0, SEEK_END);
    long size = std::ftell(fp);
    std::fseek(fp, 0, SEEK_SET);
    std::vector<uint8_t> buf(size);
    std::fread(buf.data(), 1, size, fp);
    std::fclose(fp);
    sim.writeDDR(DDR_ADDR + 0x10000 * (i - 2), buf.data(), buf.size());
  }

  uint32_t outDdr = 0;
  uint32_t outBytes = 0;

  for (const Insn &insn : insns) {
    const auto &f = insn.fields;
    if (insn.op == "SYNC") {
      sim.push(Instruction{OpCode::SYNC, 0});
      continue;
    }
    const uint32_t descAddr = hexVal(f.at("desc"));

    if (insn.op == "DMA_LOAD" || insn.op == "DMA_STORE") {
      DMAParam desc{};
      desc.sramAddr = hexVal(f.at("sram"));
      desc.ddrAddr = hexVal(f.at("ddr"));
      desc.rows = intVal(f.at("rows"));
      desc.cols = intVal(f.at("cols"));
      desc.srcStride = intVal(f.at("srcStride"));
      desc.dstStride = intVal(f.at("dstStride"));
      sim.writeDDR(descAddr, &desc, sizeof(desc));
      sim.push(Instruction{insn.op == "DMA_LOAD" ? OpCode::DMA_LOAD
                                                 : OpCode::DMA_STORE,
                           descAddr});
      if (insn.op == "DMA_STORE") {
        outDdr = desc.ddrAddr;
        outBytes = desc.rows * desc.cols * DTYPE_SIZE;
      }
    } else if (insn.op == "MATMUL") {
      MatmulParam desc{};
      desc.dstAddr = hexVal(f.at("dst"));
      desc.rhsAddr = hexVal(f.at("rhs"));
      desc.lhsAddr = hexVal(f.at("lhs"));
      desc.M = intVal(f.at("M"));
      desc.N = intVal(f.at("N"));
      desc.K = intVal(f.at("K"));
      desc.accumulate = intVal(f.at("acc"));
      sim.writeDDR(descAddr, &desc, sizeof(desc));
      sim.push(Instruction{OpCode::MATMUL, descAddr});
    } else if (insn.op == "ELEMENTWISE_ADD") {
      EwiseAddParam desc{};
      desc.dstAddr = hexVal(f.at("dst"));
      desc.rhsAddr = hexVal(f.at("rhs"));
      desc.lhsAddr = hexVal(f.at("lhs"));
      desc.n = intVal(f.at("n"));
      sim.writeDDR(descAddr, &desc, sizeof(desc));
      sim.push(Instruction{OpCode::ELEMENTWISE_ADD, descAddr});
    } else if (insn.op == "ACT") {
      ActParam desc{};
      desc.dstAddr = hexVal(f.at("dst"));
      desc.srcAddr = hexVal(f.at("src"));
      desc.n = intVal(f.at("n"));
      desc.kind = static_cast<ActKind>(intVal(f.at("kind")));
      sim.writeDDR(descAddr, &desc, sizeof(desc));
      sim.push(Instruction{OpCode::ACT, descAddr});
    } else {
      std::fprintf(stderr, "eclipse-run: unknown opcode '%s'\n",
                   insn.op.c_str());
      return 2;
    }
  }

  sim.run();

  if (outBytes > 0) {
    FILE *fp = std::fopen(outPath, "wb");
    if (!fp) {
      std::fprintf(stderr, "eclipse-run: cannot open '%s'\n", outPath);
      return 2;
    }
    std::fwrite(sim.cmodel().ddr(outDdr), 1, outBytes, fp);
    std::fclose(fp);
  }
  return 0;
}
