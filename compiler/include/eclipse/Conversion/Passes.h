#ifndef ECLIPSE_CONVERSION_PASSES_H
#define ECLIPSE_CONVERSION_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace eclipse {

#define GEN_PASS_DECL
#include "Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "Passes.h.inc"

} // namespace eclipse
} // namespace mlir

#endif // ECLIPSE_CONVERSION_PASSES_H

