#!/bin/bash

export LLVM_DIR=/home/serana/mlir/llvm-project/install/lib/cmake/llvm
export MLIR_DIR=/home/serana/mlir/llvm-project/install/lib/cmake/mlir

export ECLIPSE_NPU_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

function Eclipse-build() {
    echo "[EclipseNPU] Starting build..."
    pushd "${ECLIPSE_NPU_ROOT}/build" > /dev/null
    cmake ..
    cmake --build . -j$(nproc)
    popd > /dev/null
    echo "[EclipseNPU] Build finished."
}

function Eclipse-clean() {
    echo "[EclipseNPU] Cleaning build directory..."
    rm -rf "${ECLIPSE_NPU_ROOT}/build"
    mkdir "${ECLIPSE_NPU_ROOT}/build"
    echo "[EclipseNPU] Clean done."
}