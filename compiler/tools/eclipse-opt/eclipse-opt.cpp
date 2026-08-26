#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "eclipse/Allocation/Passes.h"
#include "eclipse/Conversion/Passes.h"
#include "eclipse/Dialect/Eclipse/EclipseDialect.h"
#include "eclipse/Emit/Passes.h"

int main(int argc, char **argv) {
  mlir::eclipse::registerPasses();
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return mlir::eclipse::createEclipseAllocate();
  });
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return mlir::eclipse::createEclipseToEasm();
  });
  mlir::bufferization::registerBufferizationPasses();

  mlir::DialectRegistry registry;
  registry.insert<mlir::eclipse::EclipseDialect, mlir::arith::ArithDialect,
                  mlir::func::FuncDialect, mlir::linalg::LinalgDialect,
                  mlir::memref::MemRefDialect, mlir::tensor::TensorDialect>();

  mlir::arith::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::linalg::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::tensor::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(
      registry);

  return mlir::asMainReturnCode(mlir::MlirOptMain(
      argc, argv, "Eclipse NPU optimizer driver\n", registry));
}
