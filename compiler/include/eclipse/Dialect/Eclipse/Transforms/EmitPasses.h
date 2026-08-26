#ifndef ECLIPSE_EMIT_PASSES_H
#define ECLIPSE_EMIT_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace eclipse {

#define GEN_PASS_DECL
#include "EclipseEmitPasses.h.inc"

} // namespace eclipse
} // namespace mlir

#endif // ECLIPSE_EMIT_PASSES_H
