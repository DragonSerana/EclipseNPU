#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "eclipse/Conversion/Passes.h"
#include "eclipse/Dialect/Eclipse/EclipseDialect.h"

int main(int argc, char **argv) {
  mlir::eclipse::registerPasses();

  mlir::DialectRegistry registry;
  registry.insert<mlir::eclipse::EclipseDialect, mlir::arith::ArithDialect,
                  mlir::func::FuncDialect, mlir::linalg::LinalgDialect,
                  mlir::memref::MemRefDialect>();
  return mlir::asMainReturnCode(mlir::MlirOptMain(
      argc, argv, "Eclipse NPU optimizer driver\n", registry));
}
