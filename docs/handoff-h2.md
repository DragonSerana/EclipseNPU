# EclipseNPU H1→H2 交接（新对话前置提示）

## 开场白（复制发给新窗口）

> 读一下 /home/serana/EclipseNPU/docs/handoff-h2.md，这是上个对话的完整交接文档（项目状态、约定、H2 计划都在里面）。确认没问题后开始 H2：先讲思路再写代码，第一步是定义 6 个与 ISA 一一对应的 Eclipse leaf op。

---

## 项目一句话

从零做 NPU 工具链：ISA → 编译器（MLIR）→ 算子 → simulator/cmodel，最终在自研模拟器上跑 Qwen2.5-0.5B。北极星是 MLIR 编译器栈（linalg → Eclipse dialect → ISA），手写代码全部是 golden 靶子。仓库 `/home/serana/EclipseNPU`。

## 已完成：H1 全部验收项（已提交，工作区干净）

- `runtime/include/eclipse_isa.h`：ISA 合同（内存映射常量 + OpCode + 4 个 descriptor struct）
- `runtime/include/dtype.h`：fp16/bf16 位级转换（RNE、次正规、inf/NaN），`loadU16/storeU16`
- `runtime/include/cmodel.h` + `lib/cmodel.cpp`：六指令（DMA_LOAD/STORE、MATMUL、ELEMENTWISE_ADD、ACT、SYNC）+ `computeCycles`；MATMUL 语义 = fp16 输入 → fp32 块内累加 → fp16 一次写回；`accumulate` 0=覆盖/1=累加；含 inplace 重叠 assert、DMA 16B 对齐 assert
- `runtime/include/simulator.h` + `lib/simulator.cpp`：指令队列 + 顺序执行 + cycle 记账（`writeDDR/push/run/totalCycles/cycleLog/cmodel()`）
- `tests/matmul_golden.cpp`：手写指令流 128×128，`int tile` 参数化 K 分块（2/4/8 全过），A 列窗口/B 行窗口 DMA 抠块，descriptor 每指令独立槽位（`nextDescAddr`）
- `tests/matmul_check.py`：对拍脚本，golden 双路径（torch fp16 优先，无 torch 退 numpy fp64）
- `docs/spec/isa-v0.1.md`、`docs/spec/accuracy.md`、`docs/cycle-report-h1.md`
- 最近提交：`a2d3902 add cycle report`（accuracy.md 实测表 + cycle 报告）

对拍结果：不分块与 torch fp16 逐位一致（0.0）；分块 2/4/8 vs fp64 = 3.9/5.8/8.4e-4，全 < 1e-2；实测按 √N 增长，随机游走误差模型实证成立。

cycle 模型参数（eclipse_isa.h 里）：`MAC_PER_CYCLE=256`、`DMA_BYTES_PER_CYCLE=32`、`DMA_FIXED_OVERHEAD=16`、`DMA_BURST_BYTES=16`、`ELEM_PER_CYCLE=128`。DMA cycle = 逐行 16B 突发块跨度计数 → ceilDiv(突发×16, 32) + 16；MATMUL = ceilDiv(M×N, 256)×K。`ceilDiv` 在 cmodel.h。

## 约定（必须遵守）

1. **改用户代码前先列改动清单征得同意**；用户对代码掌控欲强。测试/构建失败：产品代码的错要问，自己的 throwaway 脚本可以自己修
2. 命名 LLVM/MLIR 风格：类型 CamelCase（`DMAParam`/`MatmulParam`/`EwiseAddParam`/`ActParam`）；函数/变量 camelCase（`readFP16`/`computeCycles`）；常量 ALL_CAPS（`SRAM_ADDR`）；成员尾下划线（`sram_`/`ddr_`）为避同名方法
3. 头文件 Doxygen 注释（`///` + `@param`/`@return`），实现不重复注释；注释少而有用、像人写的，不要 AI 味
4. 文档中文；descriptor 字段名是 ISA 合同，改名必须同步 `docs/spec/isa-v0.1.md`
5. `n` 是元素数（非字节）；stride 是字节；内存小端、地址 16B 对齐由编译器保证
6. SYNC 在串行模型 = 0 cycle；并发调度后由调度器拉齐（木桶效应），`computeCycles` 保持纯函数
7. clang-format（LLVM 风格）保持绿；`env.sh` 提供 `Eclipse-build`/`Eclipse-format`/`Eclipse-format-check`

## H2 计划（docs/roadmap.md）

主线：
- 定义 6 个 Eclipse leaf op（与 ISA 一一对应），linalg.matmul (tensors) → tile/fuse → bufferize → convert-to-scf → convert-linalg-to-eclipse → emit ISA
- done：编译器产出的指令流与 H1 手写指令流语义等价（同一 matmul 过 PyTorch 对拍）；lit 测试接入主构建（`add_subdirectory(tests)` 已开，`tests/CMakeLists.txt` 现有 matmul_golden 目标）+ CI
- 关键参考：`tests/matmul_golden.cpp` 是编译器输出的 golden 靶子；`Simulator` 是编译器产物的消费方；H1 指令流 = 语义等价标准答案

辅线（每周 20%）：
- 9070 XT：ROCm/gfx1201 环境验证，手写 HIP fp16 GEMM（WMMA），与 PyTorch 对拍 + roofline 报告

## 环境

- 构建：`cmake --build build -j`（root CMakeLists 已开 tests 子目录；MLIR 已 find_package）
- 对拍：`cd tests && python3 matmul_check.py`（torch 2.13.0+cpu 已装：`pip install --user --break-system-packages torch --index-url https://download.pytorch.org/whl/cpu`）
- LLVM/MLIR 源码：`/home/serana/mlir/llvm-project`（命名/风格规范以其中 `llvm/docs/CodingStandards.rst` 为准）
- 工具链：Ubuntu、g++ 15、C++17、clang-format 在 `/home/serana/mlir/llvm-project/install/bin/clang-format`

## 工作风格

- 中文交流；先讲思路再写码；新增接口要讲用法；概念问题要讲透（用户会追问"为什么"，如 K 分块动机、浮点结合律、SYNC 周期语义）
- 本文件是临时交接物，新对话开始后用户会删除，不提交 git
