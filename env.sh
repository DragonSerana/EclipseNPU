#!/bin/bash

export LLVM_DIR=/home/serana/mlir/llvm-project/install/lib/cmake/llvm
export MLIR_DIR=/home/serana/mlir/llvm-project/install/lib/cmake/mlir

export ECLIPSE_NPU_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PATH="${ECLIPSE_NPU_ROOT}/build/bin:${PATH}"

export ECLIPSE_OPT="${ECLIPSE_NPU_ROOT}/build/bin/eclipse-opt"
export ECLIPSE_RUN="${ECLIPSE_NPU_ROOT}/build/bin/eclipse-run"

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

function Eclipse-compile() {
    if [[ $# -ne 2 ]]; then
        echo "usage: Eclipse-compile <input.mlir> <output.mlir>" >&2
        return 1
    fi
    input="$1"
    output="$2"
    echo "[EclipseNPU] Compiling ${input} -> ${output}"
    "${ECLIPSE_OPT}" \
        --one-shot-bufferize="bufferize-function-boundaries" \
        --convert-linalg-to-eclipse \
        --eclipse-allocate \
        "${input}" -o "${output}"
    echo "[EclipseNPU] Compile finished."
}

# 编译模型：.mlir -> .easm（完整 pipeline，含 layout 可选 bump|golden-mirror）
function Eclipse-easm() {
    if [[ $# -lt 2 ]]; then
        echo "usage: Eclipse-easm <input.mlir> <output.easm> [layout=bump|golden-mirror]" >&2
        return 1
    fi
    input="$1"
    output="$2"
    layout="${3:-bump}"
    echo "[EclipseNPU] Compiling model ${input} -> ${output} (layout=${layout})"
    "${ECLIPSE_OPT}" \
        --one-shot-bufferize="bufferize-function-boundaries" \
        --convert-linalg-to-eclipse \
        --eclipse-allocate="layout=${layout}" \
        --eclipse-to-easm="output-easm=${output}" \
        "${input}" -o /dev/null
    echo "[EclipseNPU] Compile finished."
}

# 推理：跑 eclipse-run 执行 .easm，产出结果 raw
function Eclipse-inference() {
    if [[ $# -lt 3 ]]; then
        echo "usage: Eclipse-inference <input.easm> <out.raw> [in.raw ...]" >&2
        return 1
    fi
    easm="$1"
    out="$2"
    shift 2
    echo "[EclipseNPU] Running inference on ${easm} -> ${out}"
    "${ECLIPSE_RUN}" "${easm}" "${out}" "$@"
    echo "[EclipseNPU] Inference finished."
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

# python3 tests/golden/matmul_check.py --M 16 --N 16 --K 16 \
#   --bias tests/data/bias.raw \
#   --driver "build/bin/eclipse-opt --one-shot-bufferize=bufferize-function-boundaries --convert-linalg-to-eclipse --eclipse-allocate --eclipse-to-easm=output-easm=m.easm tests/test.mlir && build/bin/eclipse-run m.easm {c} {a} {b} {bias}"