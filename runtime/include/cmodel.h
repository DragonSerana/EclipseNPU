#ifndef CMODEL_H
#define CMODEL_H

#include "dtype.h"
#include "eclipse_isa.h"
#include <vector>

namespace eclipse {

/// 对芯片的模拟：DDR/SRAM 内存模型 + 指令执行
class CModel {

private:
  std::vector<uint8_t> ddr_, sram_;

public:
  CModel();

  /// 执行一条指令
  /// @param inst 指令（opcode + descPtr，descriptor 在 DDR 命令队列区）
  void exec(const Instruction &inst);

  /// 取 DDR 地址对应的字节指针
  /// @param addr DDR 地址（DDR_ADDR 起的偏移）
  /// @return 该地址的字节指针，越界时 assert
  uint8_t *ddr(uint32_t addr);

  /// 取 SRAM 地址对应的字节指针
  /// @param addr SRAM 地址（SRAM_ADDR 起的偏移）
  /// @return 该地址的字节指针，越界时 assert
  uint8_t *sram(uint32_t addr);

  /// 从 SRAM 读一个 fp16 元素并解码为 float
  /// @param addr 元素字节地址（含偏移）
  /// @return 解码后的 fp32 值
  float readFP16(uint32_t addr);

  /// 把 float 舍入（RNE）为 fp16 写入 SRAM
  /// @param addr 元素字节地址（含偏移）
  /// @param v 要写入的值
  void writeFP16(uint32_t addr, float v);

  /// 估算一条指令的执行周期数
  /// @param inst 指令
  /// @return 周期数
  uint64_t computeCycles(const Instruction &inst) const;
};

} // namespace eclipse

#endif