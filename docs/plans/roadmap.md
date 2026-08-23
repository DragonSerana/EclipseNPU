# EclipseNPU Roadmap v0.4

> 目标：从零完成一颗 NPU（ISA → 工具链 → 算子 → simulator/cmodel），在自研模拟器上跑通 Qwen2.5-0.5B，并在真硬件（AMD RDNA4，9070 XT）上验证同一套编译/tiling 方法论。
> 方向取舍：主攻编译器 + 算子。通信不设为主线目标；多卡是 LLM 里程碑之后的远期可选项。
> 验收原则：每一步的数值验证一律与 PyTorch 参考对拍。里程碑只有同时满足验收标准和产出物才算 done。
> v0.3 变更：GPU 辅线并入主线节奏（不再后置）；LLM 里程碑保留但 done 标准写死；CNN 降为可选项；多卡降为远期可选项。
> v0.4 变更：H3 与 H4 之间插入 H3.5（单 decoder layer 端到端）——直接上全模型风险太大，先用一层把集成问题全部暴露一遍；CNN 维持可选项，不进主线。

## 指导原则

核心技能锚点：MLIR lowering 栈（linalg / transform / memref / scf）、tiling 与调度、roofline 分析、ISA 协同设计、HIP/CK 算子调优。

任何一块工作，必须落在以下三处之一：

1. 一层 IR 变换 + lit 测试
2. 一个可测量的数字——性能计数器或 roofline 位置（simulator 或真卡，占峰值的百分比）
3. 一份合同更新——ISA spec（docs/spec/isa-v0.1.md → v0.2）或算子审计（docs/ops-audit.md）

落不进任何一处的，砍掉。

手写 kernel 是 golden 参考（规格书），不是交付物；交付物是能生成等价实现的编译器。GPU 线上的手写 HIP kernel 对未来的 MLIR 后端起同样的作用。

节奏：无外部 deadline，用里程碑和产出物代替时间表。主线 : GPU 辅线 ≈ 80 : 20，辅线的 20% 是每周固定时间，不允许被主线占用。

## H1 — ISA v0.1 完整可对拍

- cmodel 实现全部六条指令（DMA_LOAD / DMA_STORE / MATMUL / ELEMENTWISE_ADD / ACT / SYNC）
- MATMUL 按正确 fp16 语义实现：half → fp32 块内累加 → half 写回（禁止对原始字节做整数运算）；accumulate 两种模式都支持（0 = 覆盖，1 = 累加）
- 跨 K-block 的 fp16 累加误差文档化（已知限制，v0.2 由 fp32 ACC 区解决）
- simulator 实现 `computeCycles()`（MAC 吞吐 + DMA 带宽模型）；cycle 模型是一等交付物，不是附属品
- 手写指令流跑通 128×128 matmul（K 维分块 + accumulate），与 PyTorch 对拍（fp16 相对误差 < 1e-2）
- Done：对拍通过；cycle 计数可输出；误差模型文档齐备
- 产出物：matmul 对拍脚本 + 第一份 cycle 报告

## H2 — 工具链主链路 + GPU 辅线（并行）

主线：

- 定义 6 个 Eclipse leaf op（与指令一一对应）
- lowering 链从 linalg 起步，StableHLO 后置：linalg.matmul (on tensors) → tile/fuse → bufferize → convert-to-scf → convert-linalg-to-eclipse → emit ISA
- 先打通一条路径（linalg.matmul → Eclipse → 指令流），正确性优先；StableHLO 留作可选练习
- lit 测试接入主构建（打开 add_subdirectory(tests)）；仓库公开 + CI（build + lit）
- Done：编译器产出的指令流与 H1 手写指令流语义等价（同一 matmul 通过 PyTorch 对拍）；CI 绿
- 产出物：完整 lowering 链 + lit 测试 + 架构说明

辅线（9070 XT，每周 10–20%）：

- 第一步只做环境验证：ROCm 对 gfx1201 的支持、`rocminfo`、最小 HIP kernel 跑通。避开已知坑：Triton gfx12 pipelining 需 `num_stages=1`；PyTorch CK SDPA 对 gfx1201 的修复较新
- 手写 fp16 GEMM：tiling + 共享内存 + WMMA（RDNA 矩阵核；与 CDNA MFMA 不同）
- Done：GEMM 与 PyTorch 对拍通过；相对峰值吞吐的 roofline 报告
- 产出物：HIP kernel（未来 MLIR 后端的 golden 参考）+ 真卡 roofline 报告

cycle 模型与 roofline（H2 与 H4 之间随进度推进）：

- profiling：DMA 带宽利用率、MAC 利用率
- 双缓冲实验用 transform dialect 表达（不写死 C++ pass）
- 量化 padding 浪费 → v0.2 硬件 padding 决策的输入
- Done：matmul 在 simulator 上的 roofline；瓶颈定性为带宽受限或算力受限

## H3 — 算子审计 → ISA v0.2 → 三算子

- 算子审计：先选模型，再定 v0.2 的 op 列表
  - 模型：Qwen2.5-0.5B（RMSNorm + SwiGLU + GQA + RoPE；备选 TinyLlama 1.1B）
  - 产出 docs/ops-audit.md：模型算子 → ISA op → 缺口 → 决策
  - 必须覆盖：嵌入层（token 依赖地址）、KV cache、RoPE、RMSNorm、SwiGLU、softmax
- 冻结 ISA v0.2，至少补齐审计发现的缺口：
  - EWISE_MUL（SwiGLU）、DIV/RSQRT（RMSNorm）、sin/cos LUT（RoPE）
  - REDUCE（max / sum / 平方和）、EXP/LUT、fp32 ACC 区 + MOVER、DMA_LOAD_ASYNC + WAIT tag（双缓冲）
  - DDR 扩到 2GB（fp16 权重约 1.4GB）；int8 量化后置到 fp16 全链路对拍通过之后，避免精度问题污染集成问题
  - 记录 per-input 静态编译限制：token 依赖地址（嵌入行、KV cache）在编译期烘焙进 DMA descriptor；间接寻址留 v0.3
- Matmul / argmax / attention 三算子
  - 每个算子两版：手写 golden + 编译器生成，两版互相及与 PyTorch 对拍
  - EclipseAttention：flash-attention 式 tiling（QK^T → softmax → PV）
  - Done：三算子对拍通过；编译器生成版性能 ≥ 手写 golden 的 90%；ISA v0.2 合同 + ops-audit 齐备
  - 产出物：ops-audit.md + ISA v0.2 + "v0.1→v0.2 每条变更由哪个模型算子驱动"的总结

## H3.5 — 单 decoder layer 端到端（H4 的探针）

- 一层 transformer 全链路对拍
  - 范围：RMSNorm → QKV projection → attention（GQA + RoPE + KV cache 写入）→ 残差 → RMSNorm → SwiGLU MLP → 残差；prefill 8 + decode 2 token
  - 目的：在小规模上把 H4 的集成问题全部暴露一遍——权重转换（safetensors → 自家格式）、多算子调度与 SRAM 编排、softmax 数值稳定性、token 依赖地址烘焙、分算子 cycle 拆解
  - H4 的通过判据：扩到全部层数后只剩"堆量"问题，不再出现新类型的问题
  - Done：与 PyTorch fp16 单层参考对拍通过（容差在开始时写入文档，H4 沿用）；输出分算子 cycle 拆解报告
  - 产出物：单层对拍脚本 + 权重转换工具链 + 分算子 cycle 拆解报告

## H4 — LLM 端到端

- 验收标准在开始时写死：
  - 模型 Qwen2.5-0.5B；prefill 128 + decode 16 token
  - logits/输出与 PyTorch fp16 参考对拍，容差在里程碑开始时写入文档
  - 全模型 cycle 报告
- 权重转换：safetensors → 自家格式
- 成本预核算：每 token ≈ 2×参数量 FLOP（0.5B 模型约 1 GFLOP/token）；标量仿真每 token 几十秒，prefill 128 + decode 16 一夜可出；目标是正确，不是吞吐
- 产出物：端到端集成证据（编译器输出 → ISA → 模型权重 → 正确输出）

## H5 — GPU 深潜：双后端闭环

- CK/rocBLAS 调优：双缓冲、swizzle、bank conflict、WMMA 极限、async copy
- MLIR ROCDL CodeGen 路径：H2 的 linalg 栈直接生成 HIP kernel，与 H2 手写 golden 对拍
- Done：MLIR 生成的 GEMM 在 9070 XT 上达到手写 golden 的 90%+；双后端（simulator vs. RDNA4）roofline 对比报告
- 产出物：双后端对比报告——核心面试叙事

## 可选项（主线之外）

- CNN（toy CNN → resnet50）：可选项。与日常工作的边缘视觉推理相关，但与目标岗位相关性低；LLM 主线跑通后按兴趣选做。
- 多卡（原 Phase 5）：远期可选项，H1–H4 完成后、与 H5 并行尝试——N 个 simulator 实例 + REMOTE_LOAD/REMOTE_STORE + 信箱/中断，两卡跑分片 MLP 与单卡对拍。定位是补全体系认知，不算主线交付物。注意：AMD 侧无 NCCL（对应 RCCL），真硬件通信经验需另找途径。

## 与目标岗位的对照

| 里程碑 | 覆盖的岗位要求 | 产出物 |
| ------ | -------------- | ------ |
| H1 | 细节严谨 / ISA co-design | fp16 语义修正 + 误差模型 + cycle 报告 |
| H2 主线 | 编译器（MLIR） | lowering 链 + lit + CI |
| H2 辅线 | 算子（HIP、roofline） | 手写 GEMM + 真卡 roofline |
| H3 | co-design | ops-audit + ISA v0.2 |
| H3.5 | 集成预演 | 单层对拍 + 分算子 cycle 拆解 |
| H4 | 全栈集成 | LLM 对拍 + cycle 报告 |
| H5 | 硬件的物理极限 | 双后端对比报告 |
