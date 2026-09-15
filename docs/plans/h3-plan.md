# H3 计划

> 依据：docs/roadmap.md 的 H3 段 + H2 实际完成情况（2026-09 复核）。
> H3 一句话：审计 Qwen2.5-0.5B 需要哪些算子 → 定 ISA v0.2 → 把 matmul/argmax/attention 三个算子做成"手写 golden + 编译器生成"两版并对拍。
> 模型参数：hidden 896 / 24 层 / 14 头（GQA 2 KV 头）/ head_dim 64 / intermediate 4864 / vocab 151936。

## 1. 关键结论（决定 H3 工作量的几个推演）

逐算子过了一遍 Qwen2.5-0.5B 的结构，roadmap 里 v0.2 的占位清单比实际需要的大，能砍掉一半：

| 模型算子 | 结论 | 对 ISA v0.2 的意思 |
| --- | --- | --- |
| 嵌入查表 | 地址静态烘焙后就是逐行 DMA_LOAD（128 行 × ~44 cycle ≈ 5.6k cycle，正确性优先够用） | 不需要新 op，GATHER 延到 v0.3 |
| RoPE | cos/sin 表宿主预计算放 DDR；rotate_half 拆成前后两半连续块，4×EWISE_MUL + 2×EWISE_SUB/ADD | 不需要 GATHER，推 v0.3 |
| down_proj（K=4864） | 超过 MATMUL 的 K≤1024 上限，必须 K 分块，跨块 fp16 累加精度问题第一次真的出现 | 先做实验：K-tile=1024 分 5 块对拍 torch，过了就把 fp32 ACC+MOVER 延到 v0.3 |
| QKV/O/gate/up（N=1152~4864） | 权重矩阵 2~8.5MB 远超 512KB SRAM，必须 N 分块（N-tile=64 时 352KB，放得下） | ISA 不用改，但编译器要从只切 K 升级成 M/N/K 三维都能切，这是 H3 编译器最大的活 |
| lm_head（N=151936） | N 分块 + 每块算局部 argmax 再合并 | 决定 REDUCE_ARGMAX 的设计 |
| Attention（seq≤128） | scores 128×128×2=32KB 整个放得进 SRAM，两遍 softmax 就行，不需要 flash 式分块；scale=0.125 在 fp16 里精确，直接折叠进 Q 权重 | "EclipseAttention" 从独立工程降级成一组组合 pattern |
| Softmax | 需要 rowwise 求最大/求和，还有广播减法乘法 | REDUCE 要加 axis 说明（按行/全量），EWISE 要加 broadcast 说明 |
| RMSNorm | 求平方和、平均、rsqrt、乘 | ACT 加 RSQRT kind |
| SwiGLU silu | 拆开要 5 条指令，给 ACT 加 SILU kind 就 1 条 | ACT kind 扩宽 {EXP, RSQRT, SILU} |

所以 ISA v0.2 的最小增量：**新增 op 是 ELEMENTWISE_SUB/MUL/DIV 与 REDUCE{kind,axis}**，外加合同修订（ACT kind 扩宽、EWISE 的 broadcast、REDUCE 的 axis、MATMUL 的 transA/transB）、DDR 扩到 2GB、DMA_LOAD_ASYNC/WAIT 只冻结编码不写编译器支持。每条变更都要写清楚是哪个模型算子逼出来的。

## 2. 阶段拆分

### H3.1 算子审计 + 冻结 ISA v0.2（1.5–2 周）

1. 写 docs/plans/ops-audit.md：模型参数表 → 逐层算子分解（prefill 128 和 decode 各过一遍）→ 每个算子对应哪些 ISA op → 缺什么 → 怎么决策。第 1 节的表就是骨架；
2. 三个小实验，记录数据：
   - down_proj 按 K=1024 切 5 块的精度——**已完成**：err=6.9e-4 < 1e-2，PASS → fp32 ACC + MOVER 推 v0.3；
   - GATHER 和逐行 DMA 的 cycle 对比——**已完成**：逐行 DMA ≈9216 cycle、占比小 → GATHER 推 v0.3；
   - EXP/RSQRT 的 LUT 项数 vs 误差——**已完成**：数据见 docs/spec/accuracy.md（N≥32 时 LUT 误差已低于 fp16 舍入）；
3. 冻结 ISA v0.2：新 op 的 ODS + spec 更新（v0.1 的六条指令语义不变；新增 opcode 按族分组插入，编号会动，但 `.easm` 是文本、按名字解析，不受影响）+ "v0.1→v0.2 每条变更由哪个算子驱动"对照表 + 已知限制（token 地址静态烘焙、16×16 阵列在 M=1 时浪费 15/16、int8 后置）。

做完的标志：每个模型算子都有映射或者写明为什么延后；v0.2 合同冻结，不再回头改。

### H3.2 cmodel/simulator 升到 v0.2（1–1.5 周）

1. 新 op 的 cmodel 实现 + computeCycles + verifier + EclipseConstants 同步；DDR 扩 2GB；
   - 超越函数（EXP/RSQRT/SILU）的 cmodel 先调 libm 占位，精度合同见第 7 节；computeCycles 用吞吐参数建模，与内部算法无关；
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

## 6. isa添加
ELEMENTWISE_SUB / MUL / DIV	 已完成
ACT kind 扩宽 {EXP, RSQRT, SILU}	EXP/RSQRT 已完成（e2e 逐位一致，0 ulp）；SILU 等 H3.3 SwiGLU
MATMUL 加 transA/transB
ELEMENTWISE broadcast
REDUCE{kind, axis}
DDR 扩 2GB	已完成（DDR_SIZE=0x80000000，基址 0x80000000→0x40000000 避开 uint32 回绕）

## 7. 超越函数策略（EXP/RSQRT/SILU）：合同定精度，实现分三步

> 决策（2026-09 讨论定稿）：cmodel 是硬件定义本身（无 RTL 对标对象），超越函数的**实现算法**（LUT 大小、插值、舍入）不在合同里绑死，合同只定**精度类**；算法由实验数据冻结。参照 NVDLA cmod/RTL 单一数据源实践（deepwiki.com/nvdla/hw/6-c-model）。

### 原则

- ISA 合同写"结果与真值误差 ≤ 某界"，不写"用 LUT 实现"——实现算法是电路层决策，属于 accuracy.md 和 spec 附录，不属于 opcode 语义。
- **cycle 与算法无关**：EXP 这类固定功能单元时序是数据无关的，computeCycles 只需要吞吐合同（每 kind 一个 per-element 速率 + 一个固定开销，注释注明是假设值）。LUT 内容影响的是数值误差，不影响 cycle 公式。速率**按 kind 分档**：RELU 是 `max(0,x)`、真芯片满速，取 `ELEM_PER_CYCLE`；EXP/RSQRT 参照 NV SFU 的 **1/4 rate**（ex2.approx/rcp.approx 的误差预算白纸黑字写在 PTX 文档里），先验取 `ELEM_PER_CYCLE/4`。等 attention 的 cycle 报告出来后用数据修正。**注意**：新增 `ACT_FIXED_OVERHEAD` 会让现有 relu 基线（matmul_add_relu_128 = 12832）变化，和 `MATMUL_FIXED_OVERHEAD` 一样要重算 baseline。
- "玩具"的分界线不是调 libm，而是有没有合同 + 排期的替换计划；本节就是替换计划。

### 三步走

1. **占位（H3.2 默认路径）**：cmodel 调 libm——EXP 用 `expf`，RSQRT 用 `1.f/sqrtf`，SILU 用 expf 组合（x/(1+e^-x)），倒数/除法由 ELEMENTWISE_DIV 承担。ISA v0.2 合同写精度类，三件事缺一不可：
   - **参考真值钉死为 fp64 计算后舍入到 fp16**（不能用 torch 当真值——版本和后端会漂移；torch 只是对拍对象）；normal 域 rel err ≤ 2^-10；
   - **subnormal 域相对误差无定义**（次正规的 ulp 相对值巨大），按绝对误差或明确 flush 行为写死；
   - **溢出行为写死**（大正数输入：饱和到 65504 还是产生 inf）——ACT 是通用 op，不能只按 softmax 的输入域想。
2. **实验（H3.1 冻结前做，纯 numpy 不写 cmodel）**：候选算法 `e^x = 2^(x·log2e) = 2^n · 2^f`，f∈[0,1) 均匀 LUT + 线性插值，扫 N∈{32,64,128,256}。**已完成**（脚本 scripts/lut_experiment.py，输入穷举全部有限 fp16）：插值误差实测 = 0.06/N²（与先验吻合），fp16 输出 **N≥32 即满足 ≤1 ulp**；但**不是逐位相同**（N=64 时约 1.1% 输入差 1 ulp），所以合同写"≤1 ulp"。RSQRT 单独扫（LUT+1 牛顿后逐位一致；只查表 N=32 时 max ulp=2）。数据见 docs/spec/accuracy.md。
   - 做完：v0.2 直接冻结算法（表大小、索引算法、插值、舍入模式写进 spec）；
   - 没做完：v0.2 只冻结精度类，算法冻结降为附录任务，不阻塞 H3.2。
3. **冻结（算法定了之后）**：表由 scripts/ 下 Python 脚本生成 C 头文件（单一数据源，同步写 spec 附录）；cmodel 从 libm 占位切换为 LUT 实现；accuracy.md 记录切换前后的对拍容差变化。**spec 附录冻结和 cmodel 切换解耦**：附录冻结随 v0.2 走；cmodel 切换放 H3.4 收尾或独立实验分支——libm 在精度类内继续合法值班，不阻塞三算子。**切换时必须保留一条 exact（libm）路径**（编译期开关或运行模式）：否则将来对拍出问题，分不清是编译器错还是 SFU 近似。

### 同策略覆盖

- **RSQRT / DIV**：候选 rcp 近似 + 1 次 Newton 迭代，同样先扫精度再冻结（硬件除法器比 exp 还贵，真实 NPU 基本不造；v0.2 的 ELEMENTWISE_DIV 就是这条路的合同入口）。
- **RoPE 的 sin/cos**：已由第 1 节"宿主预计算表放 DDR"决策覆盖，不需要硬件单元。
- **SILU**：单 kind 内部仍走 exp 路径，精度合同按组合传播写清楚（sigmoid 的 exp 误差 → 乘 x 后的传播）。

## 8. LUT 表放哪：ROM（v0.2 默认）vs SRAM（v0.3 决策点）

- v0.2 默认 **ROM**：表固化在 ACT 单元里，不可寻址——占用就是 spec 附录里的一行设计常数（64 项 × 2B = 128B 硅面积），模拟器里没有它的地址，谈不上"模拟占用"。
- **SRAM 驻留（v0.3 决策点，可提前为 H3.4 实验分支）**：表由宿主预计算 → .raw → eclipse-run 按 ABI 写 DDR → 编译器在序言发一条 DMA_LOAD 常驻 SRAM → ACT 描述符加 `lutAddr` 字段。
  - **eclipse-allocate 不用重写**：表在 IR 里就是一个普通只读 SRAM buffer，走现有 bump/golden-mirror 流程零改动；走"保留区"方案也只改 2 行（bump 起点后移，和 mirror 重叠修复同款机制）。分配策略只在"策略变化"（动态分配、lifetime 重用）时才重写，"多一种 buffer"不触发。
  - 真正的代价在合同层：描述符加字段走冻结流程。代码全是零头（conversion 几十行、emit 几行、exec 几十行、loader 几行）。
  - 收益：查表读变成真实访存流，和数据读抢 bank——正好给 v0.3 的事件驱动调度器（见 roadmap"模拟器演进"节）当第一个真实用例。