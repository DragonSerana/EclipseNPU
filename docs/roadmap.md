# EclipseNPU Roadmap v0.3

> 愿景：从零完成一颗 NPU（ISA → 工具链 → 算子 → simulator/cmodel），在自研模拟器上跑通 0.7B 量级 LLM，并在真硬件（RDNA4 9070 XT）上验证同一套编译/tiling 方法论。
> 方向取舍：主攻**编译器 + 算子**；通信不设为主线目标（主线阶段不做 NCCL/RCCL/多卡，主线完成后作为远期尝试）。
> 通用验收原则：每一步的数值验证一律与 PyTorch golden 对拍；每个里程碑必须有明确 done 标准 + 一样可对外展示的产出物，才算完。
> v0.3 变更：GPU 真硬件辅线并入主线节奏（不再后置）；LLM 保留为北极星里程碑但 done 标准写死；CNN 降为可选项；多卡降为远期可选项。

## 北极星（防跑偏锚点）

核心技能锚点：MLIR lowering 栈（linalg / transform / memref / scf）、tiling 与调度、roofline 性能分析、ISA 协同设计、HIP/CK 算子调优。

**决策规则**：任何一块工作，最终必须落在以下三处之一，落不下去就是跑偏，砍掉：
1. IR 的某层变换 + lit 测试
2. 性能计数器 / roofline 上的一个数字（simulator 或真卡，占峰值百分比）
3. ISA 合同（docs/spec/isa-v0.1.md → v0.2）或 ops-audit（docs/ops-audit.md）的更新

**手写 kernel 的定位**：每个算子的 golden 靶子（规格书），不是交付物。交付物是能生成等价物的编译器。GPU 辅线上的手写 HIP kernel 同样服务于这一定位——它们是未来 MLIR 后端对拍的目标。

**节奏规则**：无外部 deadline，用里程碑产出物代替时间表。每个 H 阶段至少产出一样可对人展示的东西；主辅线时间比约 80% / 20%（辅线固定每周，不允许被主线吃掉）。

## H1（原 Phase 1）：ISA v0.1 完整可对拍

1. cmodel 六条指令全部实现（DMA_LOAD/STORE、MATMUL、ELEMENTWISE_ADD、ACT、SYNC）
   - MATMUL 按 fp16 语义修对：half → fp32 块内累加 → half 写回（禁止 uint8 整数运算）；accumulate 0=覆盖 / 1=累加 两种模式
   - 跨 K-block 的 fp16 累加误差**文档化**（v0.2 由 fp32 ACC 解决，属已知行为）
   - simulator.cpp：`compute_cycles()` 实现（MAC 吞吐 + DMA 带宽模型），cycle 计数是**核心交付物**，不是附属品
2. 手写指令流（不经过编译器）跑通 matmul 128×128 = A×B（含 K 维分块 + accumulate）
   - done: 与 PyTorch 对拍通过（fp16 相对误差 < 1e-2）；cycle 计数可输出；误差模型文档齐备
   - 产出物：手写 matmul 对拍脚本 + 第一份 cycle 报告

## H2（原 Phase 2 前半）：工具链主链路 + GPU 辅线启动（双线并行）

3. 主线：Eclipse dialect + lowering 主链路
   - 定义 6 个 leaf op（与 ISA 一一对应）
   - 主链路**从 linalg 起步**，不引入 StableHLO：linalg.matmul (on tensors) → tile/fuse → bufferize → convert-to-scf → convert-linalg-to-eclipse → emit ISA
   - 先打通一条路径（linalg.matmul → Eclipse → 指令流），正确性优先，不做优化；StableHLO 后置为可选练习
   - lit 测试接入主构建（打开 add_subdirectory(tests)），仓库公开 + GitHub Actions（build + lit）
   - done: 编译器产出的指令流与 H1 手写指令流语义等价（同一 matmul 对拍通过）；CI 绿
   - 产出物：完整 lowering 链 + lit 测试 + 架构说明

4. 辅线：9070 XT 环境踩坑 + 手写 HIP GEMM（每周 10–20%）
   - 第一步只做环境验证：ROCm 对 gfx1201 的支持、`rocminfo`、手写 HIP kernel 跑通；避开已知坑（Triton gfx12 pipelining 需 num_stages=1，PyTorch CK SDPA 对 gfx1201 的修复较新）
   - HIP 手写 fp16 GEMM：tiling + 共享内存 + WMMA（RDNA 矩阵核；注意与 CDNA MFMA 不同）
   - done: HIP GEMM 与 PyTorch 对拍通过 + 真卡 roofline 报告（占峰值百分比）
   - 产出物：HIP kernel（后续 MLIR 后端的 golden 靶）+ 真卡 roofline 报告

5. （原 Phase 2 后半，并入 H2/H4 之间随进度做）cycle 模型与 roofline
   - profiling 工具：DMA 带宽利用率、MAC 利用率
   - 双缓冲优化实验：调度用 transform dialect 脚本表达（不写死 C++ pass）
   - 量化 padding 浪费 → ISA v0.2 "硬件 padding" 的决策依据
   - done: 输出 matmul 的 roofline 分析（simulator 侧），定位瓶颈是带宽还是算力

## H3（原 Phase 3 + v0.2 冻结）：ops-audit → ISA v0.2 → 三算子

6. 模型 op 审计（**先选模型，再定 v0.2 的 op 列表**）
   - 模型定为 Qwen2.5-0.5B（RMSNorm + SwiGLU + GQA + RoPE；备选 TinyLlama 1.1B）
   - 产出 docs/ops-audit.md：模型算子 → ISA op → 缺口 → 决策 对照表
   - 审计必须覆盖：嵌入层（token 依赖地址）、KV cache、RoPE、RMSNorm、SwiGLU、softmax
7. 冻结 ISA v0.2，至少补齐审计发现的缺口：
   - EWISE_MUL（SwiGLU）、DIV/RSQRT（RMSNorm）、sin/cos LUT（RoPE）
   - REDUCE（max/sum/平方和）、EXP/LUT、fp32 ACC 区（含 MOVER）、DMA_LOAD_ASYNC + WAIT tag（双缓冲）
   - DDR 扩到 2GB（fp16 权重放得下；**int8 量化后置**到 fp16 全链路对拍通过之后，避免精度问题污染集成问题）
   - 明确"per-input 静态编译"限制：token 依赖地址（嵌入行、KV cache）在编译期烘焙进 DMA descriptor；间接寻址留 v0.3
8. Matmul / Argmax / Attention 三算子
   - 每个算子出两版：手写 golden 版 + 编译器生成版，两版对拍
   - EclipseAttention：flash-attention 式 tiling（QK^T → softmax → PV）
   - done: 三算子对拍通过；生成版性能 ≥ 手写 golden 的 90%；isa-v0.2 合同与 ops-audit.md 齐备
   - 产出物：ops-audit.md + isa-v0.2 合同 + "v0.1→v0.2 是被哪些模型算子逼出来的"总结

## H4（原 Phase 4 的 LLM）：0.7B LLM 跑通

9. 全模型推理
   - done 标准写死：模型 Qwen2.5-0.5B；prefill 128 + decode 16 token；logits/输出与 PyTorch fp16 参考对拍（容差在开始时写入 H4 文档）；输出全模型 cycle 报告
   - 权重格式转换（safetensors → 自家格式）
   - 仿真成本预核算：每 token ≈ 2×参数量 FLOP（0.5B 模型约 1 GFLOP/token），标量仿真几十秒/token，prefill 128 + decode 16 一夜可出；不追求吞吐，只追求正确
   - done: 按上述写死标准跑通 + 对拍通过 + cycle 报告
   - 产出物：全栈集成证据（编译器输出 → ISA → 模型权重 → 正确输出）

## H5（新增）：GPU 深潜 —— 双后端闭环

10. LLM 跑通之后的主战场：
    - CK/rocBLAS 调优：双缓冲、swizzle、bank conflict、WMMA 极限、async copy
    - 给 MLIR 写 ROCDL CodeGen 路径：让 H2 的 linalg 栈直接生成 HIP kernel，与 H2 手写 golden 对拍
    - done: MLIR 生成的 GEMM 在 9070 XT 上达到手写 golden 的 90%+；"同一 tiling 决策、两个后端（自研模拟器 + RDNA4）"的 co-design 对比报告
    - 产出物：双后端 roofline 对比报告（面试核心叙事）

## 可选项（主线之外）

- **CNN（toy CNN → resnet50）**：可选项。与日常工作的边缘视觉 CNN 相关，但与目标岗位（LLM 系统）相关性低；LLM 主线跑通后按兴趣选做，不计入主线。
- **多卡通信（原 Phase 5）**：**远期可选项**，LLM 主线（H1–H4）完成后、与 H5 GPU 深潜并行按兴趣尝试：N 个 simulator 实例 + REMOTE_LOAD/REMOTE_STORE + 信箱/中断，两卡跑分片 MLP 与单卡对拍。定位是扩展视野、补全体系认知，不算主线交付物；注意 AMD 侧无 NCCL（对应 RCCL），真硬件通信经验需另找途径。
- **工程卫生（随时做）**：修对 MATMUL 后再提交（或先提交再修，不留长挂 WIP）；清理 .kilo/worktrees 与废旧分支。

## 与目标岗位的对照

| 里程碑 | 覆盖的岗位要求 | 展示物 |
|---|---|---|
| H1 | 细节严谨 / ISA co-design | fp16 语义修正 + 误差模型 + cycle 报告 |
| H2 主线 | 职责 #3 编译器（MLIR） | lowering 链 + lit + CI |
| H2 辅线 | 职责 #1 算子（HIP/roofline） | 手写 GEMM + 真卡 roofline |
| H3 | 职责 #4 co-design | ops-audit + isa-v0.2 |
| H4 | 全栈集成视野 | LLM 对拍 + cycle 报告 |
| H5 | 硬件的物理极限 | 双后端对比报告 |
