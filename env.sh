#!/bin/bash

export LLVM_DIR=/home/serana/mlir/llvm-project/install/lib/cmake/llvm
export MLIR_DIR=/home/serana/mlir/llvm-project/install/lib/cmake/mlir

export ECLIPSE_NPU_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PATH="${ECLIPSE_NPU_ROOT}/build/bin:${PATH}"

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

function Eclipse-clangd() {
    if ! command -v clangd > /dev/null; then
        echo "[EclipseNPU] clangd not found in PATH"
        return 1
    fi
    local files=("$@")
    if [[ ${#files[@]} -eq 0 ]]; then
        echo "[EclipseNPU] Scanning for .h/.c/.hpp/.cpp files..."
        mapfile -t files < <(find "${ECLIPSE_NPU_ROOT}" -type f \
            \( -name "*.h" -o -name "*.c" -o -name "*.hpp" -o -name "*.cpp" \) \
            -not -path "*/build/*" -not -path "*/.git/*" -not -path "*/.kilo/*")
    fi
    for f in "${files[@]}"; do
        echo "[EclipseNPU] clangd --check ${f}"
        clangd --check="${f}"
    done
    echo "[EclipseNPU] clangd check done."
}