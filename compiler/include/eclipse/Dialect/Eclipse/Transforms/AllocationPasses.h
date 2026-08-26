#ifndef ECLIPSE_ALLOCATION_PASSES_H
#define ECLIPSE_ALLOCATION_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace eclipse {

#define GEN_PASS_DECL
#include "EclipseAllocationPasses.h.inc"

} // namespace eclipse
} // namespace mlir

#endif // ECLIPSE_ALLOCATION_PASSES_H
