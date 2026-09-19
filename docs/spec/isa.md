# Chip Architecture v0.2

> 目标模型：Qwen2.5-0.5B（RMSNorm + SwiGLU + GQA + RoPE）。逐算子映射见
> docs/plans/ops-audit.md。v0.1 的六条指令语义在 v0.2 下保持不变。

## v0.1 → v0.2 改动

    内存模型
        DDR 1G → 2G                    驱动：全部（fp16 权重约 1.14GB，1G 放不下）
        DDR 基址 0x80000000 → 0x40000000
                                       驱动：0x80000000 + 2G 跨出 uint32 回绕成 0，
                                       越界判断全变恒假；挪基址比加宽比较省事
    指令集
        MATMUL 加 transA/transB        驱动：attention 的 Q@K^T
        新增 ELEMENTWISE_SUB/MUL/DIV   驱动：SwiGLU、RoPE、RMSNorm、softmax
        ELEMENTWISE 加 rhs 读模式      驱动：RMSNorm gamma、RoPE cos/sin、softmax 减 max
        新增 REDUCE{kind}              驱动：softmax、RMSNorm、lm_head argmax
                                       （无 axis 字段：归约方向由 cols 决定，见 §7）
        ACT kind 扩为 RELU/EXP/RSQRT/SILU
                                       驱动：softmax、RMSNorm、SwiGLU
        新增 DMA_LOAD_ASYNC / WAIT      驱动：双缓冲（仅冻结编码，v0.2 不实现）

    v0.1 规划过、v0.2 决定不做的：fp32 ACC + MOVER（精度实验已达标，推 v0.3）、
    间接寻址（推 v0.3）、int8 量化（后置）、硬件 padding。详见文末。

## 异构
    暂时不考虑添加RISC-V CPU

## 内存模型
    SRAM 0x10000000 - 0x10080000 共512KB
    DDR 0x40000000 - 0xC0000000 共2G
        0x40000000 - 0x4000FFFF  命令队列区（64KB）：指令流 + descriptor，host 写入，NPU 取指
        0x40010000 - 0xBFFFFFFF  数据区：tensor 数据
    字节对齐，tensor在内存上16字节对齐，编译器保证sramAddr/ddrAddr都是16字节对齐。
    内存控制器按照16Byte突发，突发必须是16的倍数，如果地址不是16的倍数，需要突发两次，如果没有地址对齐，比如地址落在了0x0e的位置，如果后面一个数据有16byte，之前一次突发就能拿到，现在要两次
    计算指令（MATMUL/ELEMENTWISE/ACT/REDUCE）的操作数在 SRAM 中必须 packed（行连续）存储；只有 DMA 支持 stride。
    DDR中，input tensor是啥样就啥样，SRAM中必须是packed。

## 执行模型
    指令 = (opcode: u32, descPtr: u32)，定长 8 字节；descPtr 指向命令队列区中的参数结构体（descriptor）。
    SYNC 无参数，descPtr = 0。

    编译器把指令流和 descriptor 布局在命令队列区，目前为了可读性通过 struct 表示。
    Host将指令放进环形队列，Simulator从队列读取code，读取后就删除。
    指令顺序发出，但是不保证上条指令执行完毕，如果有强依赖关系，插入SYNC指令等待所有Code执行完毕。

## 数据类型
    v0.2 仅支持fp16
    fp16 dtype = 0, size = 16bit， 带符号

## layout
    MATMUL 支持 MK*KN，以及 A/B 任一侧转置（见 MATMUL 的 transA/transB）。

## 字段单位及大小端
    字节寻址，即地址表示Byte，小端序。地址为32bit

## 指令集设计
1. DMA_LOAD
    sramAddr SRAM地址
    ddrAddr DDR地址，这里地址就是要搬运的地址，如果是整个input,那就是整个input的头，如果是小块，那就是工具链算好的小块的头
    //这两个参数的目的是为了tile，比如从640x480抠出来一个8x8的tensor.
    rows 行(裁剪的小tensor行数，单位是元素)
    cols 列(裁剪的小tensor列数，单位是元素)
    srcStride 切小方块时，要跳过的长度，即原内存pitch。单位是byte
        地址(i,j) = base + i*srcStride + j *dtype_size   (i in [0,rows), j in [0,cols))
    dstStride 目标内存的pitch，packed 就是 dstStride == cols * dtype_size，紧密排列，单位是byte

2. DMA_STORE
    同DMA_LOAD

3. DMA_LOAD_ASYNC / WAIT  （v0.2 仅冻结编码，不写编译器支持）
    DMA_LOAD 的可异步版本：携带 tag 标记，不与之前的指令同步排队；
    WAIT(tag) 等到该 tag 的异步搬运完成。用于双缓冲，用计算掩盖搬运。
    opcode 编号已预留；v0.2 不实现调度，编译器暂不使用。

4. MATMUL
    dst, lhs, rhs output/input1/input2 addr
    M, N, K MKN轴的长度（字段上限1024，仅为位宽说明）
    accumulate: 0=覆盖模式，1=累加模式
    transA / transB: 0=不转置，1=转置A / 转置B
        转置时对应矩阵在描述符里按 [K,M] / [N,K] 摆布，逻辑上仍算 MxN。
        attention 的 Q@K^T 用 transB=1。

    合法性约束：一次 MATMUL 的 dst+lhs+rhs 三块 tile 必须同时驻留 SRAM；
    违反时 simulator 直接 assert（真实硬件不检查，只会静默算错）。
    cmodel 支持任意 M,N,K≤1024；16 对齐仅为性能；padding 是编译器的可选优化
    dst和src暂时不支持inplace，即输入输出共用一块内存
    cmodel 禁止 dst/lhs/rhs 任意两块重叠（比硬件合法集保守；例如 A×A 合法但被拒），
    该保守行为在 dialect verifier 保持一致。

    数值语义：块内 fp32 累加，写回 fp16；accumulate=1 读回的 dst 是 fp16。
    跨 K-block 的累加误差由软件承担（v0.3 引入 fp32 ACC + MOVER 解决；H3 已用
    down_proj K=4864 分 5 块验证，fp16 跨块累加 err=6.9e-4 < 1e-2，故 v0.2 不引入 ACC）。

5. ELEMENTWISE_ADD / ELEMENTWISE_SUB / ELEMENTWISE_MUL / ELEMENTWISE_DIV
    dst, lhs output/input1 addr，形状 [rows, cols]
    rhs input2 addr，形状 [rows', cols']（可以小一圈，见下）
    rows 输出行数
    cols 输出行宽
    rhsBlk rhs 的行内重复周期
    rhsStride rhs 每行前进多少元素（0 = 跨行广播）
    语义：dst[i] = lhs[i] op rhs[s(i)]，其中 r = i/cols、c = i%cols、
        s(i) = (rows' == 1 ? 0 : r * rhsStride) + (c % rhsBlk)
    读模式：rhs 按 [rows', cols'] 读，行内以 cols' 为周期重复、跨行前进 cols' 或原地不动。
        rhsStride = 0 就是"跨行广播"，它同时覆盖 numpy 的 size-1 广播（[1,cols]、[rows,1]）
        和 RoPE 的块重复：cos[seq,32] 对 [seq,896] 取 blk=32/stride=32，
        不必把表物化成 [seq,896]（省 28 倍 SRAM/DMA）。
    约束：rhsBlk 整除 cols、两操作数在 SRAM 里都是 packed。
    不广播时 rows'=rows、cols'=cols，退化成 dst[i] = lhs[i] op rhs[i]（v0.1 语义不变）。

6. ACT
    dst, src output/input addr
    n 长度，元素数
    kind 元素级函数（ActKind 枚举）：
        RELU, EXP, RSQRT, SILU
    RELU/SILU 是激活；EXP/RSQRT 是特殊函数（softmax/RMSNorm 的归一化数学），
    只是指令形状相同，一道按 kind 区分。kind 只往后追加，RELU=0 不变。
    SILU(x) = x * sigmoid(x)。RSQRT = 1/sqrt(x)。除法用 ELEMENTWISE_DIV。
    精度类（合同）：对真值（fp64 算完正确舍入到 fp16）normal 域相差 ≤ 1 ulp；
    subnormal 域相对误差无定义、按绝对误差或 flush 行为处理；exp 溢出按 IEEE 产生 inf。
    实现算法（LUT 大小/插值/迭代次数）不在合同里，见 docs/spec/accuracy.md。

7. REDUCE
    dst 输出值，fp16，[rows, 1]，packed
    idx 输出索引，u32，[rows, 1]；只有 ARGMAX 用，其余 kind 必须为 0
    src 输入，fp16，[rows, cols]，packed
    rows 输出行数
    cols 每行归约长度
    kind 归约种类（ReduceKind 枚举）：MAX, SUM, SQUARE_SUM, ARGMAX
    语义：dst[r,0] = fold_{c<cols} src[r,c]，逐行独立，输出恒为 [rows,1]。
    没有 axis 字段：归约方向由 cols 唯一决定（沿最后一维）。整块归约写 rows=1——
        SRAM 里操作数必须 packed，[1, rows*cols] 是合法视图，不必另设模式。
        沿 major 维（列）归约 v0.2 不支持，绕法见文末"已知限制"。
    kind 与驱动算子：
        MAX         softmax 的行内最大值（稳定化减数）
        SUM         softmax 的行内分母
        SQUARE_SUM  RMSNorm 的 mean(x^2)。折进归约的输入端，而不是先发一条
                    ELEMENTWISE_MUL(x,x)：x[128,896] 已占 224KB，再要一块 x² 就顶到
                    512KB 上限（还要放 gamma、sum 和下一步的输出）。
        没有 MIN、没有 MEAN：都没有驱动算子。mean = SUM 后接一条 ELEMENTWISE_MUL，
        标量 1/cols 走 rhsBlk=1 / rhsStride=0 的读模式（第 5 条）。
    ARGMAX 吐两个结果：dst 是值、idx 是索引。值用于跨块比较（要合并得先能比大小），
        索引是最终产物。u32 是硬约束——vocab 151936 既超过 fp16 能精确表示的整数
        上限 2048，也超过 uint16 的 65535（昇腾的 fp16 索引就卡死在 65535）。
        索引是"相对本次指令归约切片的偏移"（0 起），将来做跨块合并加 indexBase 是纯增量。
        索引 4 字节而 DMA 的 rows/cols 按 2B/元素算，搬索引数组时按 cols=2*count 表达，
        ABI 里写清楚；IR 层用什么类型表达留到实现时定。
    输出 [rows,1] 正好接 ELEMENTWISE 的列广播（rhsBlk=1 / rhsStride=1）：softmax 的
        "减 max"、RMSNorm 的"乘 1/rms"直接复用第 5 条的读模式，不引入新概念。
    数值语义（合同见 docs/spec/accuracy.md）：行内固定二叉归约 + 指令内 fp32 累加器。
        这个累加器不是 v0.3 的 fp32 ACC + MOVER——后者是跨指令、放在内存里的累加区
        （给 MATMUL 的 K 分块接力用），前者 ISA 不可见。
    tie：多个相同最值时返回第一个；NaN 视为最大（ARGMAX 返回第一个 NaN 的索引）。

8. SYNC
    无参数，descPtr = 0，表示fence all，等待所有指令执行完成

## 示例
    struct DMAParam {
        uint32_t sramAddr;
        uint32_t ddrAddr; // 模拟空间的物理地址，不是x86的malloc地址
        uint32_t rows; //rows和cols都是元素数
        uint32_t cols;
        uint32_t srcStride;
        uint32_t dstStride;
    }

    // SYNC 无参数，descPtr = 0

    struct MatmulParam {
        // 除了DMA_LOAD/STORE，其他opcode的地址都是SRAM地址
        uint32_t dstAddr;
        uint32_t rhsAddr;
        uint32_t lhsAddr;
        uint32_t M;
        uint32_t K;
        uint32_t N;
        uint32_t accumulate;
        // v0.2 新增
        uint32_t transA;   // 0 = A 正常（MK），1 = A 转置（KM）
        uint32_t transB;   // 0 = B 正常（KN），1 = B 转置（NK），attention 的 Q@K^T 用 1
    }

    struct EwiseParam {
        uint32_t dstAddr;
        uint32_t rhsAddr;
        uint32_t lhsAddr;
        uint32_t rows;      // 输出行数
        uint32_t cols;      // 输出行宽
        uint32_t rhsBlk;    // rhs 的行内重复周期
        uint32_t rhsStride; // rhs 每行前进多少元素（0 = 行广播）
    }

    struct ActParam {
        uint32_t dstAddr;
        uint32_t srcAddr;
        uint32_t n; // 元素数
        ActKind kind; // v0.2: RELU/EXP/RSQRT/SILU
        union { //给其他激活传参数用
            uint32_t extra[4];
        }
    }

    struct ReduceParam {
        uint32_t dstAddr; // 值 [rows,1] fp16
        uint32_t idxAddr; // 索引 [rows,1] u32，只有 ARGMAX 用
        uint32_t srcAddr; // [rows,cols] fp16
        uint32_t rows;
        uint32_t cols;
        ReduceKind kind; // MAX/SUM/SQUARE_SUM/ARGMAX
    }

    // 从15x32的tensor切出来15x31
    struct DMAParam loadMatmulLhs{
        sramAddr = 0x10000000;
        ddrAddr = 0x40010000;
        rows = 15;
        cols = 31;
        srcStride = 32*2;
        dstStride = 31*2;
    }

    struct DMAParam loadMatmulRhs{
        sramAddr = 0x100003B0; //loadMatmulLhs.sramAddr+15*31*2，然后再16字节对齐
        ddrAddr = 0x400103C0;
        rows = 31;
        cols = 63;
        srcStride = 64*2;
        dstStride = 63*2;
    }

    struct MatmulParam matmulParam{
        dstAddr = 0x10001300; // loadMatmulRhs.sramAddr+31*63*2，然后再16字节对齐
        rhsAddr = loadMatmulRhs.sramAddr;
        lhsAddr = loadMatmulLhs.sramAddr;
        M = 15;
        K = 31;
        N = 63;
        accumulate = 0;
        transA = 0;
        transB = 0;
    }

    struct DMAParam loadElementwiseRhs{
        sramAddr = 0x10001A70; // matmulParam.dstAddr+15*63*2，然后再16字节对齐
        ddrAddr = 0x40011340;
        rows = 15;
        cols = 63;
        srcStride = 63*2;
        dstStride = 63*2;
    }

    struct EwiseParam elementwiseAddParam{
        dstAddr = 0x100021E0;
        rhsAddr = matmulParam.dstAddr;
        lhsAddr = loadElementwiseRhs.sramAddr;
        rows = 15;
        cols = 63;
        rhsBlk = 63;
        rhsStride = 63;
    }

    struct ActParam actParam {
        dstAddr = 0x10002950;
        srcAddr = elementwiseAddParam.dstAddr;
        n = 15*63;
        kind = Relu(0);
    }

    struct DMAParam storeActDst{
        sramAddr = actParam.dstAddr;
        ddrAddr = 0x40011AC0;
        rows = 15;
        cols = 63;
        srcStride = 63*2;
        dstStride = 63*2;
    }

    // matmul+elementwise_add+relu
    // [15*31]*[31*63] = [15*63] -> [15*63] + [15*63] = [15*63] -> Relu([15*63]) = [15*63]
    // v0.2 不排 pipeline
    // 指令流放在命令队列区（0x40000000起），定长8字节，descriptor 也在命令队列区内
    0x40000000: DMA_LOAD  loadMatmulLhs
    0x40000008: DMA_LOAD  loadMatmulRhs
    0x40000010: SYNC      // 保证两个DMA_LOAD完毕
    0x40000018: MATMUL    matmulParam
    0x40000020: SYNC      // 保证MATMUL计算完毕
    0x40000028: DMA_LOAD  loadElementwiseRhs
    0x40000030: SYNC      // 保证bias搬运完毕
    0x40000038: ELEMENTWISE_ADD elementwiseAddParam
    0x40000040: SYNC      // 保证ELEMENTWISE_ADD计算完毕
    0x40000048: ACT       actParam
    0x40000050: SYNC      // 保证ACT计算完毕
    0x40000058: DMA_STORE storeActDst
    0x40000060: SYNC      // 保证DMA_STORE完毕

## cycle模型
    当前 cycle 模型只包含 DMA 突发、MAC 吞吐、SIMD 吞吐；bank 冲突、多端口并行、惩罚周期等微架构细节留到后续性能模型
    ELEMENTWISE_* 与 REDUCE 的元素级部分走 SIMD（ALU）引擎，= ceil(rows*cols / ELEM_PER_CYCLE)；
    REDUCE 每一行是独立的归约，最后一拍的尾巴不能和下一行拼（EWISE 没有行边界，可以跨行打包）：
    = rows * ceil(cols / ELEM_PER_CYCLE) + rows * LOG2(ELEM_PER_CYCLE)
    （cols 是 128 的整数倍时和上面那种写法相同；不是整数倍时每行式略贵，贵的正是行尾那些空 lane）
    树不跨行流水（一行走完树再开下一行）是保守假设；做成流水线的话是 ceil(rows*cols/128) + LOG2 + rows 的量级；
    ARGMAX 的元素级部分 ×2（比较 + 选择，系数是假设值，待校准）；
    ACT 的 RELU 也走 SIMD（= ceil(n / ELEM_PER_CYCLE)）；EXP/RSQRT/SILU 走 SFU，
    = ceil(n / SFU_ELEM_PER_CYCLE) + ACT_FIXED_OVERHEAD，速率是假设值（等 attention cycle 报告校准）；
    MATMUL 转置与不转置开销一致。
    两个未定的假设值：REDUCE 的元素级部分是 fp32 累加器，是否仍按 ELEM_PER_CYCLE 收钱
    （fp32 通路可能只有一半 lane）；树深 LOG2(ELEM_PER_CYCLE) = 7。
    "烧哪个引擎"是可选的：列规约既能伪装成 MATMUL 的 K 循环（烧 MAC 阵列），也能用逐行
    累加（烧 SIMD）——比较两条路的周期数之前，先确认它们烧的不是同一个引擎。

## v0.2 不做 / 推迟到 v0.3

- fp32 累加区 ACC + MOVER：down_proj 跨块 fp16 累加精度已达标（err=6.9e-4），推迟到 v0.3。
- DMA_LOAD_ASYNC + WAIT：opcode 编号已冻结，编译器调度不实现，留 v0.3 做双缓冲。
- 间接寻址：token 依赖地址（嵌入行、KV cache 位置、RoPE cos/sin 偏移）全部 per-input
  静态编译、烘焙进 DMA descriptor；间接 DMA 寻址留 v0.3。
- 硬件 padding 到 MAC 倍数：依赖 roofline 实验数据决策。
- uint8/int8 量化：fp16 全链路对拍通过之后再上。
- MATMUL 的 inplace/多指令融合：后续版本。
- 沿 major 维（列）归约：没有驱动算子，不设独立模式（归约方向由 cols 唯一决定）。
  两条不改 ISA 的绕法：
    1) ones@src：`out[1,C] = ones[1,R] @ src[R,C]`，即 MATMUL(M=1, K=R, N=C)。
       被归约的维天然是 K，不需要 transpose；cmodel 的 MATMUL 内部本来就是 fp32
       累加，精度好。代价是烧 MAC 阵列：当前模型 ceil(M*N/256)*K，[128,896] 上
       512 cycle；M=1 在 16×16 阵列上浪费 15/16，H3.5 若改成 ceil(M/16)×ceil(N/16)
       就变 7168 cycle。
    2) 逐行累加：`acc[1,C] += src[r,:]`，r=1..R-1，共 R-1 条 ELEMENTWISE_ADD
       （dst 与 lhs 同一块；cmodel 的 EWISE 是逐元素读-写，原地安全）。走 SIMD 引擎，
       [128,896] 上 127*ceil(896/128) = 889 cycle，与阵列模型无关。代价是 fp16 累加
       R-1 次舍入（精度比 ones@src 差四个数量级）、约 683KB 的 SRAM 往返、R-1 条指令的
       发射开销，且 lowering 要支持带 offset 的 subview。
  两条路在当前模型下差不多，H3.5 的阵列模型一细化就反转——所以"哪个更快"取决于烧哪个
  引擎，不是孤立执行的 cycle 数（v0.3 的 no-load/no-store 调度器就是让 MAC 阵列和 SIMD
  并行，那时这个选择还会再变）。
  真加 axis 字段的代价：第二条数据通路（累加扫描，不是树）+ 新 cycle 公式 +
  ARGMAX×MAJOR 的组合语义 + verifier + 测试；换来的吞吐跟绕法 2 打平（896 vs 889），
  真正买到的是指令数 1 vs R-1 和 SRAM 流量 115KB vs 683KB。等有驱动算子再加。
