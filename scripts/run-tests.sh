#!/usr/bin/env bash
# 在 eclipse-ci 容器里编译 + 跑三关测试。由 GitHub Actions ci.yml 调用。
set -ex

which cmake && which ninja

cmake -S . -B build \
  -DMLIR_DIR=/usr/local/lib/cmake/mlir \
  -DLLVM_DIR=/usr/local/lib/cmake/llvm \
  -G Ninja

cmake --build build -j"$(nproc)"
cmake --build build --target check-eclipse check-golden check-accuracy
