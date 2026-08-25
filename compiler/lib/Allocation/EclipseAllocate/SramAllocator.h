#ifndef ECLIPSE_ALLOCATION_SRAMALLOCATOR_H
#define ECLIPSE_ALLOCATION_SRAMALLOCATOR_H

#include <cstdint>

namespace mlir::eclipse {

class SramAllocator {
public:
  explicit SramAllocator(uint32_t base, uint32_t size);

  /// 分配一块 SRAM，返回起始地址；失败返回 0。
  uint32_t allocate(uint32_t size, uint32_t alignment);

private:
  uint32_t base_;
  uint32_t size_;
  uint32_t next_ = 0;
};

} // namespace mlir::eclipse

#endif // ECLIPSE_ALLOCATION_SRAMALLOCATOR_H
