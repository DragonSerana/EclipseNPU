#ifndef ECLIPSE_DIALECT_ECLIPSEOPS_H
#define ECLIPSE_DIALECT_ECLIPSEOPS_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"

#include "EclipseOpsEnums.h.inc"

#define GET_OP_CLASSES
#include "EclipseOps.h.inc"

#endif // ECLIPSE_DIALECT_ECLIPSEOPS_H
