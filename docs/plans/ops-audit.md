# Qwen2.5-0.5B 算子审计（H3.1）

> 依据：docs/plans/h3-plan.md + docs/plans/roadmap.md 的 H3 段。
> 目的：把"Qwen2.5-0.5B 需要哪些算子"回答成"每个算子映射到哪个 ISA op、缺口是什么、怎么决策"，据此冻结 ISA v0.2。
> 范围：prefill（seq=128）为主，decode（seq=1）在每节末尾单独标注对 ISA 的影响。

## 0. 模型配置

| 项 | 值 | 备注 |
| --- | --- | --- |
| hidden_size H | 896 | |
| num_hidden_layers L | 24 | |
| num_attention_heads | 14 | |
| num_key_value_heads | 2 | GQA，比值 7 |
| head_dim d | 64 | 14×64=896（Q），2×64=128（K/V） |
| intermediate_size I | 4864 | SwiGLU |
| vocab_size V | 151936 | |
| rms_norm_eps | 1e-6 | |
| rope_theta | 1e6 | |
| tie_word_embeddings | **待核实**（见 §5） | 影响 lm_head 是否与 embedding 共享权重矩阵 |

### 权重体积（fp16，2B/元素）

| 矩阵 | 形状 | 字节 | 说明 |
| --- | --- | --- | --- |
| W_embed | [151936, 896] | ~260MB | 若未 tie，占独立 260MB |
| W_qkv | [896, 1152] | ~1.97MB | Q=896 + K=128 + V=128 |
| W_gate | [896, 4864] | ~8.31MB | |
| W_up   | [896, 4864] | ~8.31MB | |
| W_down | [4864, 896] | ~8.31MB | |
| W_lm   | [896, 151936] | ~260MB | 若 tie 则复用 W_embed |

- 每层 MLP ≈ 8.31×3 = 24.9MB；×24 层 ≈ 598MB。
- 每层 QKV ≈ 1.97MB；×24 ≈ 47MB。
- 若 untie：lm_head+embed ≈ 520MB。
- **总权重 ≈ 1.14GB（fp16）** → DDR 必须扩到 2GB（当前 v0.1 是 1GB，`eclipse_isa.h DDR_SIZE=0x40000000`）。这点与已冻结的 v0.2 规划一致。

## 1. 逐算子 ISA 分解（prefill seq=128）

下面每个算子都标：**算子 → 需要的 ISA op → 缺口 → 决策**。

### 1.1 嵌入层（embedding，token → hidden）

- 输入 token_ids[seq]，输出 hidden[seq, 896]。
- 每个 token 取 W_embed 的一行（896×2=1792B，行与行之间在内存里连续，pitch=1792）。
- **ISA op**：对每个 unique token 一次 `DMA_LOAD`，`rows=1, cols=896, srcStride=1792`。
- **地址**：token id 属于 per-input 信息，编译期已知 → 直接烘焙进 descriptor。
- **缺口**：无新 op（v0.1 的 DMA_LOAD 够用）。
- **决策**：**GATHER 推到 v0.3**，先用逐行 DMA_LOAD。cycle 数据见 §3.1。

### 1.2 QKV 投影

- 输入 hidden[seq, 896]，输出 q/k/v 拼接 [seq, 1152]。
- `[seq, 896] @ [896, 1152]`，M=seq，N=1152，K=896。W_qkv ≈ 1.97MB，**超过 512KB SRAM**。
- **ISA op**：MATMUL，需 N 分块。lhs [seq, 896] 跨 N-tile 驻留（只 load 一次）。
- **缺口**：编译器从"只切 K"升级到"M/N/K 三维都能切"（这是 H3 编译器最大的活）。
- **决策**：N_tile=64 → dst tile `128×64×2=16KB`，lhs `128×896×2=229KB`，rhs tile `896×64×2=114KB`，共 359KB < 512KB。
- 分块后每层需要 N 切 1152/64=18 块。

### 1.3 RoPE

- 作用在 Q[seq, 14, 64] 和 K[seq, 2, 64] 的最后一维 64。
- cos/sin 表宿主预计算放 DDR（shape [seq, 32]，因为是 d/2=32 对频率，广播到所有 head）。
- rotate_half：`out[..d/2] = x1*cos − x2*sin`，`out[..d/2:] = x1*sin + x2*cos`。
- **ISA op**：4× EWISE_MUL + 2× EWISE_SUB/ADD + 广播 cos/sin（[seq,32] 广播到 [seq, head, 64]）。
- **缺口**：v0.1 只有 ELEMENTWISE_ADD；需要 **EWISE_MUL**、**EWISE 的 broadcast 说明**、以及减法（或用 ADD 加负数标量，见 §4 决策）。
- **决策**：**不需要 GATHER**。x1/x2 是连续 dim 轴的前后半段，属于连续内存块，直接做元素级乘加即可。GATHER 推 v0.3。
- 每 token 的 cos/sin 地址也是 per-input 烘焙（位置 → 频率表的偏移）。

### 1.4 Attention

- Q[seq, 14, 64]，K[seq, 2, 64]，V[seq, 2, 64]；GQA 比值 7（query head h 用 kv head h//7）。
- **scores**：对每个 q-head，`Q[seq,64] × K^T[64,seq] → [seq, seq]`。需要 K^T（转置 layout）。
- **scale** = 1/√64 = 0.125，在 fp16 里**精确**（2 的幂），直接折叠进 Q 权重。
- **softmax**（两遍，scores 128×128×2=32KB 整个放得进 SRAM，不需要 flash 分块）：
  1. 按行 max → 广播减；
  2. EXP → 按行 sum → 倒数 → 广播乘。
- **PV**：`[seq, seq] @ V[seq, 64] → [seq, 64]`。
- **ISA op**：MATMUL、EWISE_MUL（scale）、**REDUCE（按行 max / 按行 sum，带 axis 说明）**、广播减/乘、**UNARY（EXP、RECIP）**、MATMUL（PV）。
- **缺口**：REDUCE{kind=MAX/SUM, axis=rowwise}、UNARY{EXP, RECIP}、EWISE broadcast、MATMUL 的 transpose layout（K^T）。
- **决策**：**"EclipseAttention"降级成一组组合 pattern**（QK^T → rowwise max → 广播减 → EXP → rowwise sum → 倒数 → 广播乘 → PV），不引入独立 attention 指令。KV cache 用静态地址 DMA（decode 时每步追加）。

### 1.5 RMSNorm（pre/post）

- 输入 x[seq, 896]，按最后一维归一化。
- 公式：`x / sqrt(mean(x²)+eps) * gamma`。
- **ISA op**：
  - x²：EWISE_MUL（x*x）；
  - 按行 sum：REDUCE{SUM, rowwise}；
  - /896、+eps、sqrt、倒数：UNARY{RSQRT}；
  - x × (1/√(mean+eps)) × gamma：EWISE_MUL + 广播（gamma[896]）。
- **缺口**：REDUCE{SUM, rowwise}、UNARY{RSQRT}、EWISE broadcast。
- **决策**：UNARY 加 RSQRT kind。

### 1.6 SwiGLU（MLP）

- gate = `x @ W_gate[896,4864]`；up = `x @ W_up[896,4864]`；silu(gate)×up → [seq,4864]；down = `[seq,4864] @ W_down[4864,896]`。
- **ISA op**：
  - gate/up 两个 MATMUL，K=896，N=4864（N 分块）；
  - silu：UNARY{SILU}（`x·σ(x)`，一条指令）；
  - ×：EWISE_MUL；
  - down：MATMUL，**K=4864 > 1024 上限，必须 K 分块**，且 N=896。
- **缺口**：UNARY{SILU}、EWISE_MUL。
- **决策**：silu 拆成 5 条指令 vs UNARY{SILU} 一条 → 选后者。

### 1.7 down_proj（K=4864 的量"第一次真正出现"）

- `[seq, 4864] @ [4864, 896]`，M=seq，N=896，K=4864。
- **关键：K=4864 必须切成 ≤1024 的块**，且 K+N 两维都要切（见 §4.2 SRAM 预算推演）。
- 跨块 fp16 累加**精度问题第一次真的出现** → 驱动 fp32 ACC + MOVER 的取舍（§3.2 实验）。
- **决策**：若 K 分块精度实验过大，则把 fp32 ACC + MOVER **推到 v0.3**（计划 §3 第 2 条）；否则 v0.2 就上。

### 1.8 lm_head

- 输入 hidden[seq, 896]，输出 logits[seq, 151936]。
- `[seq, 896] @ [896, 151936]`，N=151936 巨大，必须 N 分块。
- 推理只需 **argmax per row**，不需要完整 softmax。
- **ISA op**：MATMUL（N 分块）+ REDUCE{ARGMAX}。
- **缺口**：REDUCE_ARGMAX 的设计（分块流式合并：每块算局部 argmax 再合并，还是全 logits 落 DDR 整块 REDUCE）。
- **决策**：先做流式合并；卡住则退化为"全 logits 落 DDR 再整块 REDUCE"（慢但对，计划 §3 第 4 条）。

## 2. decode（seq=1）对 ISA 的额外影响

- 主要算子变成 M=1 的 **GEMV**：16×16 阵列会浪费 15/16 的 lane。
- 计划 §5 已记：这是 H3.5 开工时要再决定的事（cycle 模型改成 ceil(M/16)×ceil(N/16)，还是先只写文档）。
- decode 不新增 op 需求；唯一的 run-time 差异是 **KV cache 按步追加**（静态地址 DMA，每步烘焙新位置）。

## 3. 实验记录

### 3.1 GATHER vs 逐行 DMA（cycle 对比，用现有模型算，未写代码）

逐行 DMA（嵌入层，128 个 token，每行 1792B，16 对齐）：

- 每行：`rowBytes=1792`，`bursts = 1792/16 = 112`；
- `cycles = ceil(112×16 / 32) + 16 = ceil(1792/32) + 16 = 56 + 16 = 72`/行；
- 128 行 ≈ `128 × 72 = 9216` cycle。

若用**元素级 GATHER**（128 元素/cycle 的 SIMD 读 + 搬运）：每行 896 元素 ≈ 7 cycle，128 行 ≈ 896 cycle，且若 token 地址连续可一次读多个。

**结论**：逐行 DMA 92xx cycle 在整个 prefill（~万级）里占比小，且**嵌入行本身是连续内存**，GATHER 的"随机读"优势用不上。**GATHER 推到 v0.3**。唯一需要 GATHER 的场景是**非连续元素访问**，但 RoPE/attention 通过连续布局 + 广播都避开了（§1.3）。

### 3.2 down_proj K 分块精度（fp16 跨块累加）——已跑，PASS

- 目标：`[128, 4864] @ [4864, 896]`，K-tile=1024 分 5 块（4×1024 + 1×768），块间 fp16 累加（ISA v0.1 的 `accumulate=1` 语义，读回的 dst 是 fp16）。
- 判据与 tools/verify.py 一致：cos ≥ 0.999 且 `err = max|C−G| / max|G|` < 1e-2。
- 数据（seed=0，torch fp32 参考）：

  | 方案 | cos | err | |
  | --- | --- | --- | --- |
  | 单块 fp16（v0.1 baseline，K≤1024） | 0.999995 | 3.8e-4 | 基线 |
  | **跨块 fp16 累加（K=4864, 5 块）** | **0.999995** | **6.9e-4** | **PASS** |
  | fp32 跨块累加 | 1.000000 | 2.8e-7 | 精确但非必需 |

- **决策（已确认）**：跨块 fp16 累加 err=6.9e-4，远低于 1e-2，只比单块基线退化约 1.8×。按计划 §3 第 2 条**"过了就把 fp32 ACC + MOVER 延到 v0.3"**——实验 PASS → **fp32 ACC + MOVER 推到 v0.3，v0.2 不上**。这缩小了 v0.2 改动面。
- **注意**：K=4864 分 5 块含 1 块 768 余数，当前 `MatmulLowering` 要求 `K % tileK == 0` 否则 `llvm_unreachable`，所以编译器升级必须处理余数块（§3.4 三维 tiling 的附属项）。

## 4. ISA v0.2 变更建议（每条标注由哪个算子驱动）

| # | 变更 | 驱动算子 | 类型 |
| --- | --- | --- | --- |
| 1 | EWISE_MUL | SwiGLU、RoPE、RMSNorm | 新 op |
| 2 | REDUCE{kind, axis} | softmax、RMSNorm、lm_head | 新 op |
| 3 | ACT kind 扩宽为 {RELU, EXP, RSQRT, RECIP, SILU} | softmax、RMSNorm、SwiGLU | 合同修订（已决策，§4.1） |
| 4 | EWISE broadcast 说明 | RoPE、RMSNorm、softmax | 合同修订 |
| 5 | REDUCE 的 axis 说明 | softmax、RMSNorm | 合同修订 |
| 6 | MATMUL 加 transA/transB | attention（K^T，KV cache） | 合同修订（已决策，§4.3） |
| 7 | DDR 扩 2GB | 总权重 1.14GB | 一条宏 |
| 8 | DMA_LOAD_ASYNC + WAIT tag | 双缓冲掩盖搬运 | **仅冻结编码**，不写编译器支持 |

### 4.1 UNARY vs 扩展 ACT —— **已决策：扩 ACT 的 kind 枚举**

- 方案对比：计划 §1 原本写"新增 UNARY op{EXP, RSQRT, RECIP, SILU}"。但 ACT 与它指令形状完全相同（`dst, src, n, kind`），另起 UNARY opcode 会造成功能重叠的冗余 opcode。
- **确认决策**：把 ACT 的 kind 枚举**扩为 `{RELU, EXP, RSQRT, RECIP, SILU}`**（opcode 仍为 5，kind 只往后追加，RELU=0 不变 → v0.1 指令流保持合法），**不新增 UNARY opcode**。
- 语义说明：RELU/SILU 是**激活函数**，EXP/RSQRT/RECIP 是**元素级特殊函数**（softmax/RMSNorm 的归一化数学）——两者按指令形状看都属"元素级 unary/特殊函数"，共用一条指令、按 kind 区分，与业内 SFU（Special Function Unit，一条 SFU 算 exp/rcp/rsqrt 等）一致。文档里保留 ACT 名但注明"kind = 元素级 unary 函数（激活+特殊函数）"。
- 影响：v0.2 实际新 op 只剩 **EWISE_MUL + REDUCE** 两个。

### 4.2 down_proj 的 SRAM 预算（修正计划的一个假设）

计划写"K-tile=1024 分 5 块"。但若**只切 K 不切 N**：

- dst `[128,896]×2 = 229KB`（跨 K 累加驻留）；
- 单块 lhs `[128,1024]×2 = 256KB`；
- 单块 rhs `[1024,896]×2 = 1.75MB` → **远超 512KB**。

所以 down_proj **必须 K+N 两维都切**：N_tile=64 时

- dst `128×64×2 = 16KB`；
- lhs `128×1024×2 = 256KB`（跨 N-tile 驻留，只 load 一次）；
- rhs `1024×64×2 = 128KB`；
- 共 `400KB < 512KB`。

**结论**：M/N/K 三维 tiling 是必要项，不是可选优化。

### 4.3 MATMUL transpose layout（K^T）—— **已决策：MATMUL 加 transA/transB 标志**

- attention 需要 `Q @ K^T`。v0.1 只有 MK×KN。
- 方案对比：(a) MATMUL 加 transpose 标志；(b) 编译器在 DMA_LOAD 时把 K 转置读入 SRAM、复用现 MATMUL。
- **(b) 不可行**：v0.1 的 DMA 语义是 `元素(i,j) = base + i*srcStride + j*dtype_size`，即**拷 rows 行、每行 cols 个连续元素**——只能连续读，`cols` 是连续、`srcStride` 只跨行，**读不出列**。DMA_LOAD/STORE 无法转置。
- **确认决策**：走 **(a)**，在 `MatmulParam` descriptor 加 `transA`/`transB` 标志位（Q@K^T 用 `transB=1`），**不新增 opcode**；cycle 不变（M/N/K 与 MAC 数一致）。这与业内标准一致（cuBLAS/CUTLASS 的 transA/transB、oneDNN、TPU MXU 的 `A@B^T`；transformer 主力配置就是 NT：A 正常、B 转置，因权重常按 `[out,in]` 存）。
- 落地：cmodel / verifier / MatmulLowering 处理 `transB`。

## 5. 已知限制 / 待核实

- **tie_word_embeddings**：未确认。影响 lm_head 是否复用 embedding 矩阵（影响 DDR 权重布局，不影响算子/ISA 映射）。需用户在模型 config 里确认。
- **token/位置依赖地址静态烘焙**：嵌入行、KV cache 位置、RoPE cos/sin 偏移全部编译期烘焙进 DMA descriptor；间接寻址留 v0.3。
- 16×16 阵列在 M=1（decode GEMV）浪费 15/16 lane → H3.5 决策。
- int8 量化后置：fp16 全链路对拍通过之后再上。

## 6. 下一步（H3.1 剩余）

1. ~~跑 §3.2 down_proj K 分块精度实验~~ —— 已完成（PASS：err=6.9e-4 → fp32 ACC 延到 v0.3）。
2. ~~确认 §4.1 / §4.3 两个契约问题~~ —— 已确认（扩 ACT kind；MATMUL 加 transA/transB）。
3. 冻结 ISA v0.2：ODS + spec + "v0.1→v0.2 每条变更的驱动算子"对照表（变更清单见 §4 表，已收敛）。
