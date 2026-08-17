#ifndef SIMULATOR_H
#define SIMULATOR_H

#include "cmodel.h"
#include "eclipse_isa.h"
#include <deque>
#include <vector>

namespace eclipse {

/// 一条指令的周期明细（cycle 报告数据源）
struct CycleEntry {
  Instruction inst;
  uint64_t cycles;
};

/// 指令流执行器：排队、顺序执行、cycle 记账
class Simulator {
private:
  CModel cmodel_;
  std::deque<Instruction> queue_;
  uint64_t totalCycles_ = 0;
  std::vector<CycleEntry> log_;

public:
  /// 把 host 数据写入 DDR（输入 tensor / descriptor）
  /// @param addr DDR 地址（DDR_ADDR 起的偏移）
  /// @param buffer 源数据指针
  /// @param size 字节数
  void writeDDR(uint32_t addr, const void *buffer, size_t size);

  /// 压入一条指令到队列（编译器输出的消费接口）
  /// @param inst 指令
  void push(const Instruction &inst);

  /// 顺序执行队列中的全部指令，累加周期并记录明细
  void run();

  /// 查询累计执行周期数（可跨多次 run 累加）
  /// @return 总周期数
  uint64_t totalCycles() const;

  /// 逐指令周期明细
  /// @return 只读明细表（顺序与执行顺序一致）
  const std::vector<CycleEntry> &cycleLog() const;

  /// 访问内部 CModel，host 用它读写 SRAM 数据
  /// @return CModel 的可写引用
  CModel &cmodel();

  /// const 版本，只读访问
  /// @return CModel 的只读引用
  const CModel &cmodel() const;
};

} // namespace eclipse

#endif
