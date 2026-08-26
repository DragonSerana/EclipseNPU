#include "SramAllocator.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <cstdint>
#include <sys/types.h>

namespace mlir::eclipse {

SramAllocator::SramAllocator(uint32_t base, uint32_t size)
    : base_(base), size_(size) {
  next_ = base;
}

uint32_t SramAllocator::allocate(uint32_t size, uint32_t alignment) {
  uint32_t alignd = llvm::alignTo(next_, alignment);
  if (alignd + size > base_ + size_) {
    return 0;
  }

  next_ = alignd + size;
  return alignd;
}

} // namespace mlir::eclipse
