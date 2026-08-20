# EclipseNPU H2 移交 + 执行规划（合并版 · 唯一权威）

> 本文件合并并取代四份文档：`handoff-h2.md`（H1→H2 移交）、`h2-plan.md`、`h2-plan-2.md`、`h2-kickoff.md`。
> `handoff-2026-08-20.md` 是历史评估请求文档，保留作背景，不作为执行依据。
> 所有环境与基线事实均于 2026-08-20 晚在本机实测复核（见 §2、§11）。

## 0. 开场白（复制给新窗口）

> 读 /home/serana/EclipseNPU/docs/handoff-h2-final.md——这是 H2 的唯一权威移交+规划文件（合并了此前所有版本，其中的冲突已裁决，勿再翻旧文档）。确认理解后从 §8 第一周清单的 Day 0 开始：先修 golden 的 WAR 冒险。遵守 §3 协作约定：先讲思路再写码，改用户代码前列清单征得同意。

## 1. 项目一句话与岗位对齐

从零做 NPU 工具链：ISA → 编译器（MLIR）→ 算子 → simulator/cmodel，北极星 = 在自研模拟器上跑通 Qwen2.5-0.5B 推理；辅线 = 9070 XT（RDNA4）上用 HIP 验证同一套 tiling/roofline 方法论。

求职目标：DeepSeek 高性能计算团队（GPU/NPU 算子+编译器）。H2 直接对应的岗位能力是 **MLIR lowering 栈（linalg/memref/scf）+ 手写算子 + roofline**——这是选型一切技术路线时的最高优先级（见 D3 的裁决理由）。通信（NCCL/RDMA）不在 H2 范围，靠零成本的源码阅读线补偿，H5 前再评估。

三条纪律（roadmap v0.4）：任何工作必须落在 ①IR 某层变换 + lit 测试；②性能计数器/roofline 上的数字；③ISA 合同或 ops-audit 文档更新——落不进的砍掉。手写 kernel 是 golden 靶子不是交付物。

## 2. 当前状态（2026-08-20 晚复核）

**H1 已验收且当前仍绿**（刚重跑过）：`python3 tests/matmul_check.py` → PASS，max rel err = 6.109e-04（容差 1e-2），total = 11536 cycles。git HEAD = `3bf970f`（工作区有未提交的 Readme.md 改动和若干未跟踪 docs，见 §11 末尾）。

H1 交付物：六指令 cmodel + simulator + cycle 模型（`runtime/`）、fp16 codec（`dtype.h`，65536 穷举验证）、手写 golden 指令流（`tests/matmul_golden.cpp`，128×128×128，K=16×8 块）、ISA 合同（`docs/spec/isa-v0.1.md`）、精度模型（`docs/spec/accuracy.md`）、cycle 报告（`docs/cycle-report-h1.md`）。

编译器侧几乎是白纸（实测）：
- `compiler/include/.../EclipseOps.td` 是**空文件**，dialect 定义实际在 `compiler/lib/Dialect/Eclipse/EclipseOps.td`（只有 dialect 名，6 个 op 未定义）——双份漂移，H2.0 收敛为 include 单一来源；
- `eclipse-opt.cpp` 是 `int main(){return 0;}`；`Passes.h` 空；`tests/lit/*.mlir` 两个空文件；
- CMake 骨架可用，`build/bin/eclipse-opt`、`matmul_golden` 能构建出来。

**已知待修（H2.0 T0 处理）**：golden 的 K 循环有未保护 WAR 冒险（MATMUL 读 A/B_SRAM 与下一轮 DMA_LOAD 覆盖同 buffer 之间缺 SYNC；串行 cmodel 不暴露，但它是编译器对拍基准，必须先修）；`matmul_golden.cpp` 不查 argc、readFile 失败静默；accuracy.md 表头（vs fp64）与脚本默认 golden（torch fp16）口径不一致。

环境实测（影响技术路线的事实，全部已验证）：
- MLIR/LLVM **23.0.0git** 自建：`/home/serana/mlir/llvm-project`，commit `a67efda258fa73c7b6b915fb31b8412b800a15e9`（llvmorg-23-init-17286）——**pin 死，不追上游**；
- mlir-opt 有：`--one-shot-bufferize`、`--convert-linalg-to-loops`、`--convert-scf-to-cf`、`--cse`、`--canonicalize`、`--transform-interpreter`、`--affine-loop-unroll`、`--scf-for-loop-peeling`；
- mlir-opt **没有**：`--linalg-tile`（上游已删 legacy pass）和**任何通用 scf.for unroll pass**（只有 affine 专属和 test 用途的）→ 展开必须自己做（D8）；
- lit（python 包）未安装；FileCheck 只在 `llvm-project/build/bin/`，未进 install；
- torch 2.13.0+cpu 可用（主线对拍够用）；GitHub remote 已配置（`DragonSerana/EclipseNPU`）。

## 3. 协作约定（必须遵守）

1. **改用户代码前先列改动清单征得同意**；测试/构建失败：产品代码的错要问，自己 throwaway 脚本可以自己修；
2. **先讲思路再写码**；新增接口讲用法；概念问题讲透（用户会追问"为什么"）；
3. 命名 LLVM/MLIR 风格：类型 CamelCase；函数/变量 camelCase；常量 ALL_CAPS；成员尾下划线；
4. 头文件 Doxygen 注释（`///` + `@param`/`@return`），实现不重复；注释少而有用，不要 AI 味；
5. 文档中文；descriptor 字段名是 ISA 合同，改名必须同步 `docs/spec/isa-v0.1.md`；
6. `n` 是元素数（非字节）；stride 是字节；小端；16B 对齐由编译器保证；
7. SYNC 在串行模型 = 0 cycle，`computeCycles` 保持纯函数；
8. clang-format（LLVM 风格）保持绿；`env.sh` 提供 `Eclipse-build`/`Eclipse-format`/`Eclipse-format-check`；
9. 主线 : 辅线 = 80 : 20，辅线每周固定时段，不允许被主线占用；
10. 交付物以三桶纪律审计：连续 2 周落不进任何桶的在途工作，砍。

## 4. H2 Done 判据（写死）

主线（同时满足）：

1. `eclipse-opt` 从 linalg.matmul（on tensors）产出指令流文本 `.easm`；
2. `.easm` 经 runtime loader 进现有 `Simulator` 执行，128×128×128 输出过 `matmul_check.py`（rel err < 1e-2）；
3. **语义等价**（§5 D7 两层判据）全过：golden 镜像配置下与 golden v2 逐条 diff 相等 + cycle = 11536；且至少一个非镜像配置（tile-k=32 或 bump 分配器）过语义判据（数值 + hazard 0 违例 + cycle 差异可归因）；
4. lit 接入主构建（`check-eclipse` 目标），CI 绿（build + lit + e2e 对拍）；
5. `docs/notes/compiler-stack.md`：lowering 链路图 + pass 职责 + 决策记录（含本文件 D1–D9 的落地情况）。

辅线：9070 XT fp16 GEMM 对拍通过 + 真卡 roofline 报告。

时间盒：主线 9 周（H2.0 → H2.3），到期触发 §10 砍范围阶梯。

## 5. 设计决策（已裁决，含否决记录）

- **D1 linalg 起步、不引入 StableHLO**。H3/H4 的输入是手写 MLIR，无 StableHLO 消费方；linalg 是岗位通用语言，H5 GPU 后端（linalg → ROCDL → HIP）用同一份栈。
- **D2 tiling 在 conversion pattern 内直接生成分块循环**：剥出第 0 块（acc=0）+ `scf.for` 1..7（acc=1，无 iter_args——累加靠 MATMUL accumulate 标志发生在 C_SRAM 上）。不走 `linalg::tileLinalgOp` API，不依赖 transform dialect（后者留 H2.3 可选试点）。否决记录：legacy `--linalg-tile` 已删（实测）；tileLinalgOp API 是额外学习项且收益低。
- **D3 Eclipse dialect = 符号层 memref 操作数 op（三操作数、无返回值、destination-passing 风格）**，地址具体化推迟到 planning pass。**否决了 kickoff 版的"descriptor 字段全用 i32 attr"方案**，理由：attr 是常量，无法表达循环内随归纳变量变化的 DMA 源地址（ddrAddr = base + i·16·2），只能把展开提前到 conversion 期——IR 退化成平铺指令记录列表，memref/subview/scf/bufferize 全练不到，与岗位技能锚点（MLIR lowering 栈）直接冲突，且 H3 attention / H4 全模型的循环结构塞不进这个模子。attr 方案唯一合理的位置是"最终具体化层"，而 .easm 文本已经承担该角色，不需要中间 attr op。
  - 同时否决"matmul 返回 SSA 新值"形式：循环体内需 iter_args 接力；改为 dst 作第三操作数（与 bufferized linalg 的 outs 同构），循环体复用同一 %c。
- **D4 verifier 分层**：结构合同进 dialect verifier（形状一致性、dst packed、memory_space 正确、ACT kind 枚举、M/N/K 范围）；地址级合同（16B 对齐、SRAM 驻留 ≤512KB、三块不重叠的保守规则）在 planning pass 之后才查得到，放 `eclipse-allocate`/发射器的检查里报错。dialect 形态定稿见 §6。
- **D5 编译器输出 = 文本汇编 `.easm`**（一行一指令，opcode + descPtr + descriptor 全字段），runtime 侧 ~100 行 loader 喂 Simulator。可 diff、可 lit、可给 hazard 检查器消费。golden 同步增加同格式 trace 输出（D6）。DDR ABI 沿用 golden 布局：arg0/1/2 @ 0x80010000/0x80020000/0x80030000，对拍口径零改动。
- **D6 新增交付物：静态 hazard 检查器**（`tools/hazard_check.py`，~150 行）：解析 .easm，算每条指令的 SRAM/DDR 读写足迹（MATMUL accumulate=1 时 dst 也算读），程序序上无 SYNC 隔离的 RAW/WAR/WAW 交叉即报违例。先验证 golden P1 修复（修复前报 WAR、修复后 0 违例），再成为编译器产物的常驻验收器——把 H1"验证工具必须响"的教训工程化，同时压制 golden 漂移风险。
- **D7 等价判据分两层**：**主判据（语义等价，任何配置必须过）** = 数值（rel err < 1e-2）+ hazard 0 违例 + cycle 差异可归因（写进验收记录）；**回归锚点（调度等价）** = golden 镜像配置（tile-k=16 + golden 的 3-buffer 固定地址 + 逐指令 descriptor 槽位）下 .easm 与 golden v2 trace **逐条相等** → cycle 11536 自然成立，进 CI 作回归；另需 ≥1 个非镜像配置过主判据，证明编译器不是只会复刻 golden。否决记录：把"逐条 diff + 11536"当唯一 Done 会把编译器绑死在一份手写调度上（SRAM 复用、0x40 槽距这类 golden 私有 artifact 被当成合同），H3 的 argmax/attention 没有可模仿的 golden。
- **D8 静态展开在发射器内做**：本版 mlir-opt 无通用 scf unroll pass（实测）。发射器遍历 IR，对静态边界 `scf.for` 按迭代次数重复发射 body、归纳变量代常量、求值 subview 偏移 → 具体 ddrAddr。IR 不留 8 次迭代的膨胀副本。
- **D9 CI 版本策略在 H2.0 就定案**（pin commit 已记录进本文件；ccache / Docker 镜像 / 自托管三选一，写一行决策记录），实际 CI 接线在 H2.3。否决记录：拖到 H2.3 才发现 CI 无法复现本地 MLIR 版本就晚了。
- **D10 SYNC 保守全插**（与 golden v2 同构：DMA 后、MATMUL 后都插），正确性优先，由 hazard 检查器背书；cycle 最优调度留 H2.3 transform 试点 / H4。

## 6. Eclipse dialect 形态定稿（6 leaf op + 1 辅助 op）

memory_space 约定：**0 = SRAM，1 = DDR**。计算类 op 的操作数必须 space 0；DMA 的 DDR 侧必须 space 1（verifier 强制，错即编译错误）。所有 op 无返回值（指令镜像，数据流体现在 memref 同一性上，与 linalg outs 同构）。

```mlir
// 辅助 op：SRAM buffer（eclipse-allocate 把 memref.alloc(space 0) 重写成它，地址成为常量属性）
%a = eclipse.sram {addr = 0x10000000} : memref<128x16xf16, 0>

// DMA：rows/cols/srcStride/dstStride 全部从 memref 形状与 stride 推导（stride 字段 = memref stride × 2 字节）
// dst 必须 packed（verifier 查 dst 侧 stride == cols×2）；src 是 DDR 子视图（可带动态偏移）
eclipse.dma_load %a, %aView   : memref<128x16xf16, 0>, memref<128x16xf16, strided<[128,1]>, 1>
eclipse.dma_store %c, %C      : memref<128x128xf16, 0>, memref<128x128xf16, 1>

// MATMUL：M/N/K 从三个操作数形状推导；accumulate 是属性；三操作数全 space 0
eclipse.matmul %lhs, %rhs, %dst {accumulate = true}
  : memref<128x16xf16, 0>, memref<16x128xf16, 0>, memref<128x128xf16, 0>

eclipse.elementwise_add %dst, %lhs, %rhs : ...   // n = 形状元素积
eclipse.act %dst, %src {kind = 0} : ...          // v0.1 仅 0=RELU
eclipse.sync
```

128×128×128（tile-k=16）编译器 IR 骨架（发射前形态，含 D10 的 SYNC 布局——与 golden v2 逐条对齐的前提）：

```mlir
func.func @matmul(%A: memref<128x128xf16, 1>, %B: memref<128x128xf16, 1>, %C: memref<128x128xf16, 1>) {
  // convert-linalg-to-eclipse 顺带把函数参数 memref 类型改写为 space 1（ABI：全是 DDR）
  // eclipse-allocate 之后：%A/%B/%C 带 eclipse.ddr_addr 属性；%a/%b/%c 有固定 SRAM 地址
  %a = eclipse.sram {addr = 0x10000000} : memref<128x16xf16, 0>    // golden 的 A_SRAM
  %b = eclipse.sram {addr = 0x10008000} : memref<16x128xf16, 0>    // golden 的 B_SRAM
  %c = eclipse.sram {addr = 0x10010000} : memref<128x128xf16, 0>   // golden 的 C_SRAM
  // 第 0 块（剥出，acc=0）
  %a0 = memref.subview %A[0, 0][128, 16][128, 1]  : memref<128x128xf16, 1> -> memref<128x16xf16, strided<[128,1]>, 1>
  %b0 = memref.subview %B[0, 0][16, 128][128, 1]  : memref<128x128xf16, 1> -> memref<16x128xf16, strided<[128,1]>, 1>
  eclipse.dma_load %a, %a0
  eclipse.dma_load %b, %b0
  eclipse.sync
  eclipse.matmul %a, %b, %c {accumulate = false}
  eclipse.sync                                   // WAR 保护：下一轮 DMA 要覆盖 %a/%b
  // K 其余 7 块
  %ik = ...                                       // %i × 16（scf.for 内可静态求值）
  scf.for %i = 1 to 8 {
    %ai = memref.subview %A[0, %ik][128, 16][128, 1] : ...
    %bi = memref.subview %B[%ik, 0][16, 128][128, 1] : ...
    eclipse.dma_load %a, %ai
    eclipse.dma_load %b, %bi
    eclipse.sync
    eclipse.matmul %a, %b, %c {accumulate = true}
    eclipse.sync
  }
  eclipse.sync
  eclipse.dma_store %c, %C
  func.return
}
```

`.easm` 格式（golden trace / 编译器输出 / hazard 检查器 / diff 工具四方共用，字段序固定）：

```
DMA_LOAD   desc=0x80000100 sram=0x10000000 ddr=0x80010000 rows=128 cols=16 srcStride=256 dstStride=32
MATMUL     desc=0x80000148 dst=0x10010000 lhs=0x10000000 rhs=0x10008000 M=128 N=128 K=16 acc=1
SYNC
```

## 7. 阶段计划

### H2.0 地基（第 1 周）——见 §8 逐日清单

出口闸门：golden v2（对拍 PASS + total=11536 + trace 可 dump）；hazard 检查器对 golden v2 = 0 违例；6 op + 结构 verifier 的正负例 lit 全绿（`check-eclipse` 一键可跑）；CI pin 策略决策记录在案；EclipseOps.td 单一来源。

### H2.1 全链 spike：16×16×16 直通（第 2–3 周）

Day 7 的 bufferize 实验结论 → 决定 `convert-linalg-to-eclipse` pattern 写法。新增（每个 pass 配 lit）：
1. `convert-linalg-to-eclipse`：linalg.matmul（bufferized，单块不 tiling）→ dma_load×2 + sync + matmul + sync + dma_store；`memref.copy` → dma_load/store；顺带把函数参数类型改写为 space 1；
2. `eclipse-allocate`：`memref.alloc`(space 0) → `eclipse.sram {addr}`（16B 对齐 bump 或 golden 镜像布局，pass option 切换）；func 参数挂 `eclipse.ddr_addr`（ABI 表）；地址级合同检查（对齐/驻留/重叠）在此报错；
3. `eclipse-to-easm` 发射器：本阶段无循环，平铺发射；
4. `eclipse-run`（runtime loader）：读 .easm + .raw → 按 ABI 写 DDR → 构造指令 → `sim.run()` → dump 输出与 trace；
5. `matmul_check.py` 加 `--driver` 参数指向 eclipse-run。

出口闸门：16×16×16 的 .easm 进 Simulator 对拍 PASS + hazard 0 违例 + .easm 人可读可 diff。
风险预案：bufferize 语义卡住 > 1 周 → 入口改为已 bufferize 的 linalg on memrefs（仍是 linalg 栈，tensor 入口后补）。

### H2.2 固定 tiling 直通 128×128×128（第 4–7 周，主线 Done 主体）

1. conversion pattern 加 tile-k 分块（D2：剥首块 + scf.for）；
2. 发射器加静态展开（D8：归纳变量代常量、求值 subview 偏移）；
3. 双配置验收（D7）：golden 镜像配置逐条 diff golden v2 trace + cycle 11536；非镜像配置（bump + tile-k=32）过语义判据三层；
4. `docs/notes/compiler-stack.md` 初稿。

### H2.3 加固 + 加分（第 7–9 周）

1. elementwise 融合链：`linalg.add`/relu 的 pattern → matmul + bias add + relu 一次驻留 SRAM（C_SRAM 不落 DDR 直接接 ewise/act，即 isa spec 示例链）；
2. SRAM 分配器一般化（bump 为默认，镜像布局为 option）；
3. transform dialect 试点（可选）：`transform.structured` 表达同款 tiling，与 D2 路径 diff 一致 → H4 双缓冲实验载体；
4. CI 接线（pin commit + ccache/Docker，D9 已定案）+ 仓库公开 + README；
5. 文档定稿。

## 8. 第一周逐日清单（每天 1–2 小时）

| 天 | 内容 | 出口 |
| -- | ---- | ---- |
| Day 0 | 环境一次性配置 + T0 修 golden | golden v2 commit |
| Day 1 | T1 断言/文档清理 + T0.2 hazard 检查器 | 检查器对 golden v2 报 0 违例 |
| Day 2 | T2 lit 跑道 + EclipseOps.td 去重 + eclipse-opt 注册 | 冒烟 lit 绿 |
| Day 3–4 | T3 六个 op + eclipse.sram 的 ODS + assemblyFormat | round-trip 可 parse |
| Day 5 | T3 结构 verifier + 负例 lit | check-eclipse 绿 |
| Day 6 | T4 H2.0 出口闸门 checklist + CI pin 决策记录 | 闸门全勾 |
| Day 7 | bufferize 实验（16×16×16 手写 linalg 过 `-one-shot-bufferize`，观察 memref 形态，笔记进 compiler-stack.md） | H2.1 的 pattern 写法定型 |

**Day 0 环境配置**（~30 分钟）：

```bash
pip3 install --user lit                       # lit 当前未装
ls /home/serana/mlir/llvm-project/build/bin/FileCheck   # 确认存在（不在 install 里）
# LLVM commit 已 pin 在本文件 §2：a67efda258fa73c7b6b915fb31b8412b800a15e9
```

**T0 修 golden P1 WAR**（`tests/matmul_golden.cpp`，K 循环末尾补 SYNC；改用户代码，先列清单征得同意）：

```cpp
for(int i = 0; i < tile; i++) {
    pushDmaLoad(...); pushDmaLoad(...);
    sim.push(Instruction{OpCode::SYNC, 0});
    pushMatmulBlockTile(sim, (i==0)?0:1, 128/tile);
    sim.push(Instruction{OpCode::SYNC, 0});   // ← 新增：保护 A/B_SRAM 的 WAR
}
```

顺手：argc 检查 + readFile 失败报错退出；增加与 `.easm` 同格式的 trace dump（第 4 个 argv 指定输出路径，或固定输出 `golden.easm`）。验收：`matmul_check.py` PASS + total=11536 不变。commit：`golden v2: fix K-loop WAR hazard; add trace dump`。

**T1 断言与文档**：`eclipse_isa.h` 加 `static_assert(sizeof(Instruction)==8 / DMAParam==24 / MatmulParam==28 / EwiseAddParam==16 / ActParam==32)`；isa-v0.1.md MATMUL 段补一句"cmodel 禁止 dst/lhs/rhs 任意两块重叠（比硬件合法集保守，A×A 合法但被拒），该保守行为在 dialect verifier 保持一致"；accuracy.md 表头口径与脚本默认对齐（都写 torch fp16，fp64 作附录）。

**T0.2 hazard 检查器**：`tools/hazard_check.py`，输入 .easm，输出违例列表（指令序号 + 冒险类型 + 涉及地址区间）。自检：对修复前 golden（临时还原一版 trace）应报 8 处 WAR，修复后 0 处——这一正一反就是它的验收测试。

**T2 lit 跑道**：`tests/lit/lit.cfg.py`（`ShTest(False)`，suffixes=[".mlir"]，替换符 `%eclipse-opt`=build/bin/eclipse-opt、`%mlir-opt`=install/bin/mlir-opt、`%FileCheck`=llvm-project/build/bin/FileCheck）；`tests/CMakeLists.txt` 加 `check-eclipse` 目标（`find_program(LLVM_LIT lit REQUIRED)` + custom_target，DEPENDS eclipse-opt matmul_golden）；冒烟测试用 upstream pass（canonicalize 常量折叠）证明跑道通。

**T3 op 定义**：ODS 按 §6 形态；verifier 清单：

| op | 结构检查（dialect verifier） | 地址检查（allocate/发射器） |
| -- | -- | -- |
| dma_load/store | dst/src space 正确；dst packed（stride==cols×2）；形状 2 维 | 地址 16B 对齐；srcStride ≥ cols×2 |
| matmul | 三操作数 space 0；形状 lhs(M,K)/rhs(K,N)/dst(M,N) 一致；1≤M,N,K≤1024 | 三块 16B 对齐、驻留 SRAM、两两不重叠（保守） |
| elementwise_add/act | 操作数 space 0、形状一致；kind==0 | 对齐 + 驻留 |
| sync / sram | 无 | sram addr 对齐且区间在 SRAM 内 |

lit 正例：6 op + sram 各一条 parse 通过；负例（expected-error）：M=1025、sramAddr 未对齐、dst==lhs 重叠、越出 SRAM、act kind=1、DMA src space=0。空文件 `add_op.mlir` 并入 leaf_ops.mlir 后删除。

## 9. 辅线（9070 XT，每周固定 10–20%，2–3 个月日历）

- **G1 环境验证（1–2 周）**：rocminfo/rocm-smi、gfx1201 支持矩阵、hipcc 最小 kernel（vector add）；从第一天维护 `docs/notes/gfx1201.md`（ROCm 版本 pin、已知坑：Triton gfx12 pipelining 需 `num_stages=1`、CK SDPA 对 gfx1201 修复较新、rocminfo 输出摘录）。注意当前 torch 是 cpu 版，真卡参考环境在 G1 一并确认（ROCm torch 或 rocBLAS）；
- **G2 手写 fp16 GEMM（3–6 周迭代）**：naive → shared memory tiling（128×128×16，与主线同参数，方法论互通）→ WMMA（RDNA 矩阵核，注意与 CDNA MFMA 的文档差异，gfx1201 支持需实测）→ bank conflict/swizzle/double buffer 逐步加；每步与 rocBLAS/PyTorch 对拍；
- **G3 roofline 报告（1–2 周）**：rocBLAS/hipBLASLt 最大 fp16 TFLOPS 作实测峰值参考、arithmetic intensity 曲线、自家 GEMM 定位 → `docs/roofline-rdna4.md`。这是面试硬数字锚点（模拟器 vs 真卡占峰值百分比）的前半。

## 10. 降级阶梯与风险

时间盒到期按序砍：① transform 试点 → ② elementwise 融合（EWISE/ACT 编译器路径挪 H3 补，matmul-only 保底）→ ③ 非镜像配置验收（退回镜像逐条等价的验收强度）→ ④ 底线不可砍：golden 修复 + hazard 检查器 + 16×16×16 spike + 128 固定 tiling 直通 + lit。128 也卡住 → 保 spike 链路 + 文档化差距，绝不无限延期。

风险：R1 LLVM 版本漂移（pin commit，不追上游）；R2 bufferize inplace 倾向 vs ISA"dst/src 不共用"合同（lowering 从不把 linalg 结果直接写进 DDR 参数——引 SRAM tile + dma_store，allocate 后置检查兜底）；R3 lit/FileCheck 路径（Day 0 解决）；R4 CI 构建成本 40–60 分钟（ccache/Docker/自托管，Day 6 定案）；R5 ROCm/gfx1201 未知数（G1 先行，辅线绝不阻塞主线）；R6 golden 漂移（hazard 检查器 + 镜像 diff 进 CI）；R7 时间摆动（H2 是全项目最大摆动项 ±2 月，阶梯兜底）；R8 无 scf unroll pass（发射器内展开）；R9 .td 双份漂移（Day 2 收敛）；R10 编译器过拟合 golden（D7 两层判据 + 非镜像配置）。

## 11. 环境速查

- 构建：`cmake --build build -j`（root CMakeLists 已开 tests 子目录；MLIR 已 find_package）；`env.sh` 提供 `Eclipse-build`/`Eclipse-format`/`Eclipse-format-check`；
- 对拍：`python3 tests/matmul_check.py`（torch 2.13.0+cpu 已装；golden 双路径 torch fp16 优先）；
- mlir-opt：`/home/serana/mlir/llvm-project/install/bin/mlir-opt`；FileCheck：`/home/serana/mlir/llvm-project/build/bin/FileCheck`；clang-format：`/home/serana/mlir/llvm-project/install/bin/clang-format`；
- LLVM/MLIR 源码（命名/风格/参考模板）：`/home/serana/mlir/llvm-project`（toy 教程、GPU dialect 的 memref 型 op 是 §6 op 的参考）；编码规范以其 `llvm/docs/CodingStandards.rst` 为准；commit `a67efda258fa`（23.0.0git）；
- 工具链：Ubuntu、g++ 15、C++17；
- git：remote `git@github.com:DragonSerana/EclipseNPU.git`；当前 HEAD `3bf970f`，Readme.md 有未提交改动，`docs/` 下五份规划/移交文件未跟踪——开工前建议：本文件提交入库，四份被取代文件按惯例删除（用户操作，AI 不要代删）。
