# EclipseNPU 项目规划 v0.1

> 愿景：从零完成一颗 NPU（ISA → 工具链 → 算子库 → simulator/cmodel），最终在其上跑通 0.7B LLM。
> 通用验收原则：每一步的数值验证一律与 pytorch golden 对拍；每步有明确 done 标准才算完。

## Phase 1: ISA + cmodel/simulator

1. 冻结 ISA v0.1（architecture.md 写完示例后冻结），实现 runtime
    - runtime/include/eclipse_isa.h：Instruction struct + OpCode 枚举
    - runtime/lib/simulator.cpp：SRAM/DDR 内存模型 + 6 条指令解释执行，**内置 cycle 计数**（MAC 吞吐 + DMA 带宽模型）
    - 先手写指令流（不经过编译器）跑通 matmul 128×128 = A×B（含 K 维分块 + accumulate）
    - done: 手写指令流与 pytorch 对拍，误差 < 1e-2；cycle 计数可输出

## Phase 2: 工具链

2. Eclipse dialect + lowering 主链路
    - 定义 6 个 leaf op（与 ISA 一一对应）
    - 主链路：StableHLO → linalg → bufferize(memref) → scf tiling → convert-linalg-to-eclipse → emit ISA
    - 先打通一条路径（linalg.matmul → Eclipse → 指令流），正确性优先，不做优化
    - done: 编译器产出的指令流与 Phase 1 手写指令流语义等价（同一 matmul 对拍通过）

3. matmul 端到端 + 对拍
    - 任意 shape（非 MAC 倍数 → 编译器 padding）、K 维分块、accumulate
    - 自动化：lit 测试 + pytorch golden 脚本
    - done: 形状参数化的 matmul 全部对拍通过

4. cycle 模型与 roofline 分析
    - profiling 工具：DMA 带宽利用率、MAC 利用率
    - 量化 padding 浪费 → 触发"硬件 padding"TODO 的决策依据
    - 双缓冲（SYNC 粒度细化）作为优化实验
    - done: 输出 matmul 的 roofline 分析，定位瓶颈是带宽还是算力

## Phase 3: 算子库（对标 cuBLAS/CUTLASS/DeepGEMM）

4.5 算子库
    - EclipseMatmul(ctx, A, B, C, params)：shape 检查 → tiling → 指令流 → push → sync
    - EclipseArgmax：移植手写 NPU 经验（ireduce/cmp/mul/fence 模式）
    - EclipseAttention：flash-attention 式 tiling（QK^T → softmax → PV）
    - done: 三个算子对拍通过，接口稳定可复用

## Phase 4: 模型

5. toy CNN → resnet50
    - toy CNN（MNIST 级）：conv（copy+matmul 即 implicit-gemm）、pool、RELU
    - resnet50：BN fold、skip connection（EWISE_ADD）、各 conv 变体
    - done: resnet50 全模型在 simulator 跑通，逐层与 pytorch 对比

6. 0.7B LLM
    - 前置约束：fp16 权重 1.4GB > DDR 1GB → 选项 A：DDR 扩到 2GB（改宏）；选项 B：先做 int8 量化（architecture TODO）
    - 权重格式转换（safetensors → 自家格式）
    - done: 0.7B 推理跑通，输出与 pytorch fp16 参考对拍通过

## Phase 5: 多卡（远期）

7. 多卡通信
    - N 个 simulator 实例 + REMOTE_LOAD/REMOTE_STORE + 信箱/中断
    - done: 两卡跑分片 MLP，输出与单卡一致
