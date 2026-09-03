#ifndef ECLIPSE_ELIDE_COPIES_PASSES_H
#define ECLIPSE_ELIDE_COPIES_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace eclipse {

#define GEN_PASS_DECL
#include "EclipseElideCopiesPasses.h.inc"

} // namespace eclipse
} // namespace mlir

#endif // ECLIPSE_ELIDE_COPIES_PASSES_H
