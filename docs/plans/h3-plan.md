# H3 计划

> 依据：docs/roadmap.md 的 H3 段 + H2 实际完成情况（2026-09 复核）。
> H3 一句话：审计 Qwen2.5-0.5B 需要哪些算子 → 定 ISA v0.2 → 把 matmul/argmax/attention 三个算子做成"手写 golden + 编译器生成"两版并对拍。
> 模型参数：hidden 896 / 24 层 / 14 头（GQA 2 KV 头）/ head_dim 64 / intermediate 4864 / vocab 151936。

## 1. 关键结论（决定 H3 工作量的几个推演）

逐算子过了一遍 Qwen2.5-0.5B 的结构，roadmap 里 v0.2 的占位清单比实际需要的大，能砍掉一半：

| 模型算子 | 结论 | 对 ISA v0.2 的意思 |
| --- | --- | --- |
| 嵌入查表 | 地址静态烘焙后就是逐行 DMA_LOAD（128 行 × ~44 cycle ≈ 5.6k cycle，正确性优先够用） | 不需要新 op，GATHER 延到 v0.3 |
| RoPE | cos/sin 表宿主预计算放 DDR；rotate_half 用 GATHER 对换 + 乘 ±1 掩码 | 决策点：GATHER（通用）还是专用 ROPE op |
| down_proj（K=4864） | 超过 MATMUL 的 K≤1024 上限，必须 K 分块，跨块 fp16 累加精度问题第一次真的出现 | 先做实验：K-tile=1024 分 5 块对拍 torch，过了就把 fp32 ACC+MOVER 延到 v0.3 |
| QKV/O/gate/up（N=1152~4864） | 权重矩阵 2~8.5MB 远超 512KB SRAM，必须 N 分块（N-tile=64 时 352KB，放得下） | ISA 不用改，但编译器要从只切 K 升级成 M/N/K 三维都能切，这是 H3 编译器最大的活 |
| lm_head（N=151936） | N 分块 + 每块算局部 argmax 再合并 | 决定 REDUCE_ARGMAX 的设计 |
| Attention（seq≤128） | scores 128×128×2=32KB 整个放得进 SRAM，两遍 softmax 就行，不需要 flash 式分块；scale=0.125 在 fp16 里精确，直接折叠进 Q 权重 | "EclipseAttention" 从独立工程降级成一组组合 pattern |
| Softmax | 需要 rowwise 求最大/求和，还有广播减法乘法 | REDUCE 要加 axis 说明（按行/全量），EWISE 要加 broadcast 说明 |
| RMSNorm | 求平方和、平均、rsqrt、乘 | UNARY 加 RSQRT |
| SwiGLU silu | 拆开要 5 条指令，给 UNARY 加 SILU kind 就 1 条 | UNARY{EXP, RSQRT, RECIP, SILU} |

所以 ISA v0.2 的最小增量：**新 op 只有 3 个左右**（EWISE_MUL、REDUCE{kind,axis}、UNARY{kind}），外加两条合同修订（REDUCE 的 axis、EWISE 的 broadcast）、DDR 扩到 2GB、DMA_LOAD_ASYNC/WAIT 只冻结编码不写编译器支持（避免以后改 opcode 编号）。每条变更都要写清楚是哪个模型算子逼出来的。

## 2. 阶段拆分

### H3.1 算子审计 + 冻结 ISA v0.2（1.5–2 周）

1. 写 docs/plans/ops-audit.md：模型参数表 → 逐层算子分解（prefill 128 和 decode 各过一遍）→ 每个算子对应哪些 ISA op → 缺什么 → 怎么决策。第 1 节的表就是骨架；
2. 两个小实验，记录数据：
   - down_proj 按 K=1024 切 5 块的精度（手写指令流 + verify.py 加 shape 参数）；
   - GATHER 和逐行 DMA 的 cycle 对比（用现有 cycle 模型算，不用写代码）；
3. 冻结 ISA v0.2：新 op 的 ODS + spec 更新（opcode 只往后追加，v0.1 的指令流保持合法）+ "v0.1→v0.2 每条变更由哪个算子驱动"对照表 + 已知限制（token 地址静态烘焙、16×16 阵列在 M=1 时浪费 15/16、int8 后置）。

做完的标志：每个模型算子都有映射或者写明为什么延后；v0.2 合同冻结，不再回头改。

### H3.2 cmodel/simulator 升到 v0.2（1–1.5 周）

1. 新 op 的 cmodel 实现 + computeCycles + verifier + EclipseConstants 同步；DDR 扩 2GB；
2. 顺手做掉一直搁置的 MATMUL_FIXED_OVERHEAD：H3 要对比"编译器生成版和手写版的 cycle"，这个 overhead 不加，小 tile 的对比不公平。加的时候在注释里写明这是模型假设；
3. hazard_check.py 认识新 opcode（现在只认六条指令，新 op 一进来就查不了）；
4. 每个新 op：lit 正反例 + 单 op 手写指令流 + 和 torch 对拍。

做完的标志：所有新 op 单测全绿。

### H3.3 三个算子（2.5–3.5 周，H3 最重的部分）

1. MatmulLowering 三维 tiling（约 300–500 行）：M×K×N 循环嵌套，K 块之间用 accumulate 接力，N 块的 dst tile 互相独立，lhs 跨 N-tile 驻留（lhs 只 load 一次，这是第一个真正的优化）。验收：down_proj 全形状跑通并对拍 torch；
2. argmax（约 1 周）：REDUCE_ARGMAX 的 cmodel + lm_head 分块流式合并的 pattern；手写 golden 版 + 编译器版，两版都对拍；
3. attention（约 1.5 周）：组合 pattern（QK^T → 按行求 max → 广播减 → EXP → 按行求和 → 倒数 → 广播乘 → PV）+ KV cache 静态地址 DMA；手写 golden 版；对拍容差开工时写进文档；
4. 每个算子：两版都过 hazard 检查器 + cycle 报告 + 编译器版 ≥ 手写版的 90%（同样调度下应该接近 100%，留 10% 是给调度差异的余量）。

做完的标志：三个算子两版对拍全过，性能线达标。

### H3.4 收尾（0.5–1 周）

cycle-report-h3（分算子的 MAC/DMA 利用率）、ops-audit 定稿、CI 绿、给 H3.5 留移交注记（token 地址烘焙、KV cache 地址管理、权重转换工具的接口这些集成风险）。

## 3. 如果卡住了，按这个顺序砍

1. GATHER 推到 v0.3，先用逐行 DMA（有数据背书）；
2. fp32 ACC + MOVER 推到 v0.3（K 分块精度实验过了就触发）；
3. attention 两遍 softmax 精度实在不行，才考虑 flash 式分块（预计用不上）；
4. argmax 流式合并卡住，就把全部 logits 落 DDR 再整块 REDUCE（慢但对）；
5. 三维 tiling 卡住，先保 K+N 两维（M 固定 128），M 分块挪到 H3.5。

## 4. 时间预估

6–9 周业余时间。比 roadmap 估的 4–8 周略长，多出来的主要在三维 tiling——从"只切一个循环"到"真正的 GEMM 调度"是个台阶，但这也是 H3 里最值钱的部分。

## 5. 记到 H3.5 门口的事

16×16 阵列波形量化问题：H3 的测试 shape 都很友好（M=128）不会暴露，但 decode 一进来主力算子就是 M=1 的 GEMV，会浪费 15/16 的 lane。H3.5 开工时要决定：cycle 模型改成 ceil(M/16)×ceil(N/16)（历史基线数字两边一起改）还是先只写文档。
