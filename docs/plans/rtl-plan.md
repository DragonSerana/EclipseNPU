# RTL 支线计划

> 定位：学习支线（流水线、握手、反压、存储系统）+ cycle 模型校准手段，**不是流片**。
> 触发（2026-09，H3.2 完成时）：cmodel 的时序是分析模型，H3.3 起算子优化要消费 cycle 数字，担心"模型太理想 → 优化错上加错"；同时借 RTL 把流水线、握手、反压这些设计侧概念从书本变成肌肉记忆。
> 与 cmodel 的关系：RTL 存在后，cmodel = 合同/golden，RTL = 实现。方向与 NVDLA 相反——NVDLA 的 RTL 是产品、cmodel 是功能参照，我们是 cmodel 为产品、RTL 去复刻它。所以 RTL 不能反过来修正 cmodel 的数值，只能暴露 cmodel 结构上缺了什么（§2）。
> 2026-09 调整：单元顺序改为先存储系统、后算术阵列；done 标准从"与 cmodel 一致"改为"cycle 能手算、波形与手算一致"；新增歧义清单（§5）作为每层固定产出；bit-exact 加法器降为可选单元 R6；插入点从"H3.3 之前"改为"与 H3.3 并行"，不设 gate。

## 1. 目标与非目标

目标：

- cycle 从公式变成状态：每个 cycle 必须有一个寄存器或一级流水为它负责。`DMA_FIXED_OVERHEAD=16`、`ACT_FIXED_OVERHEAD=8`、`REDUCE_TREE_STEPS=7`、`SFU_ELEM_PER_CYCLE=ELEM_PER_CYCLE/4` 这些值要么被证实、要么被修正。
- 握手、反压、stall 传播：反压怎么从 SRAM 端口一路传到 sequencer，为什么每个模块单看都很快、接起来掉一半吞吐。
- 存储系统——惩罚的发生地：bank 化 SRAM 的端口分配与仲裁、DMA 与 MAC 抢端口、`transB` 之后 B 矩阵的列向访问、权重流式（LLM 推理是 memory-bound，权重 DMA 是主角，见 ops-audit §7.4）。
- 频率与面积：cmodel 里没有"秒"。yosys 综合给出 f_max 和面积之后，`MAC_PER_CYCLE=256` 才能换算成 ns，峰值算力 = MAC 数 × 频率 才有物理含义，roofline 的分母才不是空的。
- bit-exact 验证流程的完整体验（R6，可选）。
- 产出：校准时序参数表 + 歧义清单（§5）。

非目标（明确放弃，省 70% 工作量）：

- 不流片、不追求时序收敛、不做时钟域跨越、不做 AXI 总线、不碰 DDR 控制器（行为级 memory model 代替）。
- 不指望 RTL 告诉你"真实商用 NPU 的惩罚有多大"——RTL 里只存在你自己设计进去的物理（§2）。惩罚的**机制**由 RTL 解释，**量级**取决于你写的存储模型。

## 2. 认知边界（防止错误期待）

- RTL 给的是惩罚的**因果**（"fill/drain 原来是从这里来的"），不是**量级**——没建模 bank conflict，你的芯片里它就不存在；SRAM 做成零延迟无限端口，你的 RTL 就永远不会 stall。
- 要量级只有两条路：读 NVDLA 的 RTL 看工业级做法（真流片、真约束），在 9070 XT 上用 rocprof 实测。
- 三角闭合才算完整：**自己写 RTL 解释机制 → 读 NVDLA RTL 看工业级做法 → rocprof 实测真实数字**。三件事都要做，单靠任何一个都有盲区。

### 2.1 业内对照（每个决策的出处，2026-09 检索核实）

| 本计划的决策/概念 | 业内先例 | 出处 |
| --- | --- | --- |
| cmodel = 合同/golden，RTL = 实现，testbench 逐项比对 | NVDLA 仓库本身就是三件套：RTL + Cmodel + testbench，RTL 输出对 cmodel 比对 | github.com/YuechengLi/hw（NVDLA hw 镜像，仓库自述"RTL, Cmodel, and testbench"）；deepwiki.com/nvdla/hw |
| LUT 表驻片上 RAM、宿主加载、硬件线性插值 | NVDLA SDP/CDP 的激活 LUT：X 表 65 项（线性/指数两种模式）+ Y 表 257 项（线性），16bit/项；软件经寄存器接口自动递增加载（要求子单元 IDLE 时更新）；越界用可编程斜率外推；带 hit/miss 优先级寄存器和统计计数器（XHitNum 等）——**h3-plan §8"表放 SRAM"决策的直接先例** | nvdla.org/hw/v1/ias/lut-programming.html |
| RISC-V 在 NPU 里的角色 | NVDLA 内置 RISC-V 核（基于 SiFive Freedom 平台的 NV_SMALL，32 位）做配置/监控子系统——**控制，不是算子计算**；和"exp 跑 RISC-V"的边缘做法是两种真实取舍 | riscv.org：NVIDIA's DLA Meets SiFive's Freedom Platform（Sijstermans & Lee, 2018） |
| 非线性双轨：近似硬件 + 精确软件 | NV GPU：SFU/MUFU（ex2.approx，误差预算印在文档）管快档；libdevice expf（软件多项式，跑普通 FMA）管准档 | h3-plan §7 已引 |
| 性能探索用解析模型，RTL 只做设计冻结后的验证 | Timeloop/Accelergy：分析模型 + 标定成本表做加速器设计空间探索，是学界/工业界标准流程；RTL 在流程末端 | timeloop.mit.edu |
| bank 化 SRAM | NVDLA CBUF：512KB 多 bank 卷积缓冲，喂 CMAC 阵列 | NVDLA hw 文档/仓库 |
| 数据流选 output-stationary 16×16（R4） | NVDLA CMAC / TPU MXU 是 weight-stationary（权重驻留阵列、特征流过）——我们有意不同：权重复用靠 DMA tiling，省阵列内广播网络。**代价是每拍 32 个元素读的 SRAM 端口带宽，以及 B 的列向访问**，正好是 R2 的题目 | NVDLA hw 仓库；TPU ISCA'17 论文 |

### 2.2 开源参考（已筛公信度：机构背书 / 活跃维护 / 有验证体系，个人项目不收）

| 项目 | 公信度 | 有什么 | 对 RTL 支线 | 对主线（编译器/算子） |
| --- | --- | --- | --- | --- |
| **Gemmini**（UC Berkeley，Chipyard 旗舰生成器） | Berkeley ASPIRE 实验室，学术引用量大，有上硅流片记录 | GEMM 脉动阵列的 Chisel RTL，**WS/OS 数据流可配**（直接对照我们 R4 的 OS 决策）；Spike 功能仿真 + FireSim 周级仿真；驱动 + 软件栈 | R4 的数据流/累加结构对照：它能两头切，我们为什么只选 OS | GEMM tiling 与编译器协同；学术上有跑 transformer 类负载的实践 |
| **VTA**（Apache TVM 基金会） | Apache 基金会项目 | 张量加速器：ISA + Chisel RTL + C++ 周期近似仿真器（virtual VTA）+ TVM 编译器全栈 | 小型"指令驱动加速器"的完整参照 | **和我们 linalg→eclipse 链最同构的"编译器+加速器协同设计"教材**（缺点：社区维护已放缓，当教材别当依赖） |
| **Vortex**（Georgia Tech） | 学院团队，活跃维护 | 开源 RISC-V GPGPU 全栈 RTL + 模拟器；有端到端 ML 支持的工作（OSCAR 论文 "Toward End-to-End ML Support on Vortex"） | GPU 形态的 warp/共享内存/分支对照 | GPU 怎么跑 LLM kernel——H5 的旁证 |
| **Ara / Snitch**（ETH Zürich PULP） | 欧洲学术旗舰 | 开源 RISC-V 向量机 RTL + 性能模型 | "非线性跑可编程向量核"的工程级参照（与我们公司的 RISC-V 路线同族） | v0.3 若加控制/向量核的定位参考 |
| Timeloop / Accelergy | MIT/Harvard/NVIDIA 系 | 解析性能/成本模型 | 校准方法论 | 设计空间探索标准流程（§2.1 已引） |
| MLPerf Inference | MLCommons 基金会 | 标准基准与规则 | — | 校准数字的公信力来源 |

读法：每进入一个 R 单元，先看 Gemmini/VTA 的对应模块（R2 看 bank 缓冲与仲裁、R4 看 PE 与累加），写三行对照笔记（我们的做法 / 它的做法 / 差异定性）——沿用 industry-mapping 方法。

排除记录（说明筛过，不是没找到）：个人 GPT-2 NPU 项目（单人、无验证体系）；Arm Ethos-U（编译器 vela 开源、**RTL 闭源**）；Tenstorrent / Groq / Cerebras / Google TPU（闭源，只有论文可读）。

## 3. 验证架构（核心资产）

- Verilator 把 RTL 编成 C++，与 cmodel 挂**同一个 test harness**：同一份指令流分别喂给 cmodel 和 Verilated RTL，在**指令边界**比对 SRAM/DDR 内容与 cycle 数。
- 比对的开销要先算清：`CModel` 构造时就 `ddr_.resize(DDR_SIZE)`（v0.2 是 2GB），TB 再链一个 CModel 就是两份 2GB 虚拟内存，全量逐字节 diff 不可行。比对按区域做（每条指令声明的 dst/src 范围）或按哈希，且 TB 侧的 DDR 尺寸要可配。
- `.easm` 解析器现在在 `tools/eclipse-run.cpp` 的匿名命名空间里（`parseEasm`），TB 复用不了。开工前先把它抽成库（放 EclipseRuntime 或单独一个小 lib）——这是 R1 的前置条件，不是"顺手"能带过的。
- 差异归因分三类，不能一句"两边必有一错"：
  1. cmodel 的假设错（例如 per-instruction 的固定开销，见 §4 R4）；
  2. **模型有意串行、RTL 真并发**：cmodel 的 DMA 就是 `memcpy`、`SYNC` 返回 0、指令由 host 直接 `push`、没有取指；RTL 一旦真做异步 DMA 与 sequencer 取指，cycle 必然系统性不等。这一类不是 bug，是 roadmap"模拟器演进"要消灭的对象，也是本支线最值钱的产出；
  3. 一边有 bug。抓 cmodel 的 bug 和抓 RTL 的 bug 同样是学习，每次不一致都是认知密度最高的一刻。
- 方法论齐装（业内标配，一样别省）：**lint 先行**（`verilator --lint-only -Wall` 过了再仿真）；**scoreboard**（cmodel 就是现成的 scoreboard，别重写）；**coverage**（`verilator --coverage`，R4 收口看行覆盖）；**性能计数器进 RTL**（每 opcode 占用拍数、每个资源的 stall 拍数）——校准表从计数器自动生成，不靠人肉数波形；**单时钟域 + 同步复位**（不做 CDC，见 §1 非目标）。
- CI 挂 check-rtl：Verilator 回归进现有 CI，失败自动留波形。要动 `Dockerfile.ci` 装 verilator，并且**把版本钉死**——Verilator 版本会改 lint 行为，不钉这个 gate 会飘。
- 保留 serial 记账基线（H1 = 11536 永可复现）；RTL cycle 数与 cmodel 各条公式出对照表，差异按上面三类逐条归因。

## 4. 单元递进 R1–R6（每层独立可停，停在哪都不浪费）

排序依据是"惩罚发生在哪"，不是算术难度。

- **R1：valid/ready 流水与反压**（1 周内）。一个带 skid buffer 的两级流水，下游加一个真实的 stall 源（比如每 4 拍只能收 1 拍）。
  done：能凭手算写出反压如何逐级传播、涉及哪些信号组合、为什么 `ready` 的组合逻辑不成环；波形与手算一致。
- **R2：bank 化 SRAM 与仲裁**。SRAM 做成 4/8 bank（字交织）+ 仲裁 FSM：sequencer 发 DMA 写的同时 MAC 在读，端口冲突在这里第一次真实发生。开工前先写下端口预算：16×16 阵列每拍需要 16 个 A + 16 个 B = **32 个元素读**，而 DMA 通道只有 16 元素/拍（`DMA_BYTES_PER_CYCLE=32`）——SRAM 的读端口和 DMA 通道是两条不同的带宽，这 2× 从哪来是 R2 要回答的第一个问题。
  done：bank 冲突 stall 在波形里可见并与手算对上；能回答"bank 数/读端口数怎么决定 `N_tile`"——即用端口预算重新推一遍 ops-audit §4.2 那个"400KB < 512KB"的账。
- **R3：DMA 引擎 + descriptor 取指**。descriptor 从 DDR 的命令队列取。
  done：LOAD/STORE 与 cmodel 逐 bit 一致；突发/对齐/固定开销模型验证或修正；**取指本身的拍数被记录下来**——cmodel 完全没有这一项（host 直接 `push`），这是 §3 第二类差异的第一笔。
- **R4：16×16 MAC 阵列 + 顶层跑通 H1 指令流**（output-stationary）。范围写死：
  - 含 `transB`（attention 的 `Q@K^T`，v0.2 已冻结 `MatmulParam` 的 transA/transB）、含 `accumulate=1`（跨 K 块，dst 是 fp16，要读改写）、含 M/N/K 不是 16 的倍数时的尾块处理；
  - 覆盖的 opcode 只有 H1 子集（MATMUL / DMA_LOAD / DMA_STORE / SYNC）。编译器还会发 REDUCE、ACT、ELEMENTWISE 变体，那些不在 R4 范围——"编译器指令流免费获得 RTL 验证"这句话到此为止并不完全成立。
  - 微架构决策在这里敲定：`MAC_PER_CYCLE=256` = 16×16 二维阵列，H3.5 门口的"波形量化/GEMV"悬案按二维解读，公式改成 `ceil(M/16)×ceil(N/16)`。这个改动对 H1 基线是 **no-op**（128 和 64 都是 16 的倍数，数字不变，11536 仍是 11536）；变的只有 decode（M=1）和 N 的尾块。
  - **fill/drain 是每个输出 tile 一份，不是 per-instruction 常数。** H2 的 K=16 分块把 128×128×128 切成 8 条 MATMUL（`tests/golden/golden.easm`），每条 64 个 tile：`8×64×(K+D)` 与 `8×64×K` 差 `512D` 拍，D=4 时 8192 → 10240（+25%）；而按一个 per-instruction 常数塞进去只加 `8D` 拍（+0.4%），把小块 K 的代价低估 64 倍。所以这里回填的是**公式的形式**（`ceil(M/16)×ceil(N/16)×(K+D)`），D 由你选的流水深度定，RTL 只负责确认 FSM 里有没有额外 stall。这件事直接决定 H3.3"编译器版 vs 手写版"的对比公不公平：模型里没有 per-tile 项，小 K 块看起来就是免费的。
  - 数值验收口径：cmodel 是 K 次串行 fp32 加（`cmodel_exec.cpp` 里 `execMatmul` 的 `acc +=`）。物理上更自然的宽累加器 + 写回只舍入一次会比 cmodel 更准，但必然不逐 bit 一致。R4 定为"相对 fp64 参考 ≤1 ulp，且误差不差于 cmodel"；要逐 bit 一致走 R6。
  done：在上述口径下与 cmodel MATMUL 一致；cycle 能按 `ceil(M/16)×ceil(N/16)×(K+D)` 手算并与波形对上；H1 的 128×128 手写指令流全链 bit 一致 + 逐指令 cycle 对照表。
- **R5（按需，随时可停）**：UNARY/LUT 单元（EXP 表，与 h3-plan §7 的 LUT 实验联动；NVDLA SDP 的 LUT 是现成参照物，见 §2.1）、EWISE 单元、**REDUCE 单元**（rowwise max/sum——加法树是和 elementwise 完全不同的一类流水结构，softmax/RMSNorm/argmax 都要用）。R5 之前，`ACT_FIXED_OVERHEAD`、`REDUCE_TREE_STEPS`、`SFU_ELEM_PER_CYCLE`、`ARGMAX_ELEM_PER_CYCLE` 都是假设值，H3.3 的 cycle 报告要标注。
- **R6（可选，随时可停）：bit-exact 加法器**。fp32 RNE 加法器（对齐移位、舍入位、粘滞位）——这是数值工作，不是微架构工作，所以从 R1 里摘出来单列。两个把恐惧砍半的事实：fp16×fp16 乘积在 fp32 里**精确**（11+11=22 位尾数 ≤24，乘法本身不引入舍入）；fp16 输入的乘积与累加**碰不到 fp32 的 subnormal 区**（最小乘积 2^-48 ≈ 3.6e-15，离 fp32 最小正规数 1.2e-38 还有 23 个数量级）——不需要处理 double rounding 和 subnormal，只要加法器 RNE 正确。配套一个 fp32→fp16 RNE 转换，65536 全空间穷举比对，golden 是现成的 `runtime/include/dtype.h`。
  done：随机 10 万对 + 定向边界（同指数、大差指数、舍入进位链）与 cmodel 逐 bit 一致。

每个单元的流程：先画 micro-arch 草图（流水线级数、握手信号、SRAM 端口安排）入 `docs/notes/design/`，再写 RTL，再进 harness 对跑。

## 5. 歧义清单（每层固定产出）

RTL 会把 ISA 和 cmodel 没定义的东西逼出来：一个信号要么在这拍有效、要么不在，没有中间状态。每遇到一条就记一条，格式固定：

```
问题 | ISA 与 cmodel 现状 | RTL 的选择 | 后续动作（改 isa.md / 改 cmodel_cycles.cpp / 记为已知限制）
```

清单本身就是交付物——它把 RTL 接回主线的"三选一"（一层 IR 变换 + lit、一个可测量的数字、一份合同更新）。开工前就能预见的几条：

1. MATMUL 的 fill/drain：per-tile 还是 per-instruction？（§4 R4，直接决定 cycle 公式的形式）
2. `accumulate=1` 时 dst 同时被 DMA 写，谁赢？cmodel 顺序执行，没有这个问题。
3. `SYNC` 等的到底是什么：DMA 写完 SRAM、写完 DDR，还是写回队列排空？
4. descriptor 取指是否计入指令周期？（cmodel 是 host `push`，完全没有这一项）
5. M/N/K 不是 16 的倍数时，阵列尾块是屏蔽还是补零；REDUCE 的越界 lane 已有说法（取最大填 -inf、求和填 0，见 microarch.md），MATMUL 侧还没有对应条款。
6. DMA 与 MAC 同时访问 SRAM 时端口怎么分（R2）；`transB=1` 时 B 矩阵按列访问，bank 冲突算在谁头上。
7. 各执行单元的速率假设值（`ACT_FIXED_OVERHEAD`、`REDUCE_TREE_STEPS`、`SFU_ELEM_PER_CYCLE`、`ARGMAX_ELEM_PER_CYCLE`）在 R5 之前都没有 RTL 背书。

## 6. 范围纪律

- 语言：SystemVerilog 可综合子集；工具：Verilator + GTKWave；**yosys 在 R2/R4 done 时各跑一次综合报告**（f_max + 面积）——把 cycle 换算成 ns 和硅面积，是"cycle 定义更明确"的临门一脚，小模块综合只要几分钟。
- **AI 分工**：数据通路样板（fp16 乘法拼接、加法树、FIFO 骨架）可以让 AI 起草；控制 FSM、握手、波形调试必须自己来——调试就是学习的本体，AI 代劳等于没买这张门票。判断标准：关掉代码，能凭记忆画出 FSM 的每个状态与转移条件、能预测下一拍每个信号的值。做不到就是 AI 写的，自己只是在运行它。
- 每次仿真之前先手算这一层的预期拍数和关键信号的预期波形，跑完拿波形对照。差异出现的地方才是学习发生的地方；AI 可以帮忙解释差异，但差异要自己发现。
- AI 生成的 Verilog "仿真过了就收下"是最大的坑（latch、多驱动、位宽截断、X 传播都可能恰好过关）：`-Wall` 警告当错误、`--assert` 常开、ROM/RAM 显式初始化。
- harness 直接吃 .easm 格式：R4 之后编译器产出的 H1 子集指令流免费获得 RTL 验证——cmodel、RTL、编译器三方对上（范围见 §4 R4）。
- 无 deadline 但有 done 标准：done = "cycle 能手算 + 波形与手算一致 + 歧义清单有新增"。不用"和 cmodel 逐 bit 一致"当通用 done——那会把人从"理解"拉到"匹配一个自己都知道不完整的模型"。

## 7. 插入点、预算与降级

- 插入点：**与 H3.3 并行**，固定每周 ≤15% 的时间，不设 gate，不阻塞主线。原先"H3.3 之前先校准"的理由站不住：H3.3 的 90% 是同一模型下的相对赛跑，fill/drain 在比值里抵消（§1）；Timeloop 那一类流程也是解析模型先做设计空间探索、RTL 在流程末端（§2.1）。本支线真正的理由是硬件设计的第一手理解 + 校准值的因果解释。
- 主线不等 RTL：算子/编译器在 cmodel 上先行；R5/R6 不阻塞任何主线。
- **前置（不是 RTL 的工作，但没有它 RTL 没有可比对象）**：先把 cycle 模型的形状改对——MATMUL 加 per-tile 的 `(K+D)` 项、wave quantization（M/N 不是阵列边长的倍数时的填充浪费）改成 `ceil(M/16)×ceil(N/16)`，再出一张假设值的敏感度表（D ∈ {0,2,4,8}、SFU 1/4 vs 1/1、`ELEM_PER_CYCLE` 128 vs 64 各自会不会翻转 H3.3 的 90% 判定）。纸面改动，半天到一天。
- 校准值的归属：R2/R3/R4 的产物回填 `EclipseConstants` 与 `cmodel_cycles.cpp` 的公式形式，落在 H3.4 的 cycle 报告里。H3.3 的相对对比不依赖校准值，但依赖模型形状正确。
- 降级/停下：每层 done 独立，停在哪都不浪费。不写"X 周没进展就降级"的硬阈值——业余时间的估计本来就不准，写了必然触发。改成"连续两个自然月没有可跑的仿真就停"。

## 8. 起步手册

### 8.0 环境（一次性，全部 apt，不源码编译）

```bash
sudo apt install verilator gtkwave yosys
verilator --version    # 5.x 即可
```

- 当前环境三个都没装（2026-09 核实），第一步先把这条跑掉。
- iverilog 不装（与 Verilator 重复）；编辑器 VS Code + mshr-H 的 SystemVerilog 插件（要的是高亮，补全靠 AI）。
- lint 用 `verilator --lint-only -Wall` 就够，verible 可选。

### 8.1 目录与构建

```
rtl/                 # 可综合源码（SystemVerilog）
  common/            # 共享参数（package：TILE_M、BANKS…）
  pip/               # R1
  bank/              # R2
  dma/  top/         # R3/R4
  mac/               # R4
tb/                  # C++ harness（Verilator），链接 EclipseRuntime
  common/            # cmodel 桥、.easm 解析（从 tools/eclipse-run.cpp 抽出来）
  pip/  bank/  mac/
```

- 先手敲命令跑通，再固化为 CMake custom target（目标名 check-rtl，接 §3 的 CI）：

```bash
verilator --cc --exe --build -j 8 --trace --assert -Wall \
  --top-module mac_array rtl/mac/*.sv tb/mac/harness.cpp \
  -CFLAGS "-Iruntime/include" --Mdir build/rtl/mac -o tb_mac
```

### 8.2 第 0 步：Hello Verilator（1–2 晚，把工具链跑热）

写一个 8 位计数器 + harness，学会三件事：

1. Verilator 的节奏：`top->clk=0; eval(); 改输入; top->clk=1; eval();`——**一个 posedge 就是 cmodel 里的一拍**，cycle 从此有物理锚点；
2. `--trace` 出 VCD → GTKWave 打开看波形——第一个"波形时刻"；
3. 写之前先 `--lint-only -Wall` 的习惯。

### 8.3 每层的节奏

先在 `docs/notes/design/rtl-<单元>.md` 画框图（端口、流水级、握手）→ AI 起草数据通路 → 自己写/调 FSM → lint → 先写预期再仿真 → 比对 → 记歧义清单 → commit。

### 8.4 cycle 的口径（第一天就定死，避免以后扯皮）

- 1 cycle = 1 个 posedge；指令占用 = 第一个输入就绪 → 末级输出寄存器写入；
- fill/drain = 乘法/累加链装满与排空的拍数，**每个输出 tile 一份**，不是每条指令一个常数（§4 R4）；
- 综合给出的 f_max 用来把 cycle 换成 ns；cmodel 侧仍然只记 cycle，ns 只出现在报告里；
- RTL cycle 与 serial 模式 cmodel 对表出差异归因（§3 三类），对照表就是校准交付物。

### 8.5 小白常见坑清单（每个都是真实时间杀手）

- 阻塞/非阻塞混用：always_comb 用 `=`，always_ff 用 `<=`；
- always_comb 少 else → 推出 latch，lint 会叫；
- 位宽不匹配：`-Wall` 全开、警告当错误；
- RTL 里不写延迟/initial（那是 TB 的事）；
- 存储器要初始化（cmodel 有 memset 语义，RTL 的 RAM 上电是 X）；
- 一个信号只允许一个 always 写（多驱动）；
- 对比失败先怀疑 cmodel——两边必有一错，一半概率错在自以为对的那边（§3 精神）。

### 8.6 与主线的接口

- R1–R4 全程不碰编译器/runtime 现有代码：harness 只链接 EclipseRuntime，`dtype.h` 就是 golden；唯一要动的现有代码是把 `.easm` 解析从 `tools/eclipse-run.cpp` 抽成库（§3）。
- 校准值（per-tile fill/drain、bank 冲突 stall、取指拍数、加法器拍数）→ 回填 `EclipseConstants` 与 `cmodel_cycles.cpp` → H3.4 cycle 报告。
- 歧义清单里的"改合同"条目 → `docs/spec/isa.md`；"改模型"条目 → cmodel；"记已知限制"条目 → 留在本文件里。
- R4 开工前读 Gemmini 的 PE/累加结构、VTA 的指令驱动结构（§2.2），写三行对照笔记（我们的做法/它的做法/差异定性）。
