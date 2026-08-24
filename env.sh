#!/bin/bash

export LLVM_DIR=/home/serana/mlir/llvm-project/install/lib/cmake/llvm
export MLIR_DIR=/home/serana/mlir/llvm-project/install/lib/cmake/mlir

export ECLIPSE_NPU_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PATH="${ECLIPSE_NPU_ROOT}/build/bin:${PATH}"

export ECLIPSE_OPT="${ECLIPSE_NPU_ROOT}/build/bin/eclipse-opt"

function Eclipse-build() {
    echo "[EclipseNPU] Starting build..."
    mkdir -p "${ECLIPSE_NPU_ROOT}/build"
    pushd "${ECLIPSE_NPU_ROOT}/build" > /dev/null
    cmake ..
    cmake --build . -j$(nproc)
    popd > /dev/null
    echo "[EclipseNPU] Build finished."
}

function Eclipse-test() {
    echo "[EclipseNPU] Running tests..."
    cmake --build "${ECLIPSE_NPU_ROOT}/build" --target check-eclipse check-golden -j$(nproc)
    echo "[EclipseNPU] Tests finished."
}


function Eclipse-clean() {
    echo "[EclipseNPU] Cleaning build directory..."
    rm -rf "${ECLIPSE_NPU_ROOT}/build"
    mkdir "${ECLIPSE_NPU_ROOT}/build"
    echo "[EclipseNPU] Clean done."
}

function Eclipse-src-files() {
    local files=("$@")
    if [[ ${#files[@]} -eq 0 ]]; then
        mapfile -t files < <(find "${ECLIPSE_NPU_ROOT}" -type f \
            \( -name "*.h" -o -name "*.c" -o -name "*.hpp" -o -name "*.cpp" \) \
            -not -path "*/build/*" -not -path "*/.git/*" -not -path "*/.kilo/*")
    fi
    printf '%s\n' "${files[@]}"
}

function Eclipse-format-check() {
    if ! command -v clang-format > /dev/null; then
        echo "[EclipseNPU] clang-format not found in PATH"
        return 1
    fi
    local files=()
    mapfile -t files < <(Eclipse-src-files "$@")
    local failed=0
    for f in "${files[@]}"; do
        if ! clang-format --dry-run --Werror "${f}" > /dev/null 2>&1; then
            echo "[EclipseNPU] NOT formatted: ${f}"
            failed=1
        fi
    done
    if [[ ${failed} -ne 0 ]]; then
        echo "[EclipseNPU] Format check FAILED. Run Eclipse-format to fix."
        return 1
    fi
    echo "[EclipseNPU] Format check passed (${#files[@]} files)."
}

function Eclipse-format() {
    if ! command -v clang-format > /dev/null; then
        echo "[EclipseNPU] clang-format not found in PATH"
        return 1
    fi
    local files=()
    mapfile -t files < <(Eclipse-src-files "$@")
    clang-format -i "${files[@]}"
    echo "[EclipseNPU] Formatted ${#files[@]} files."
}