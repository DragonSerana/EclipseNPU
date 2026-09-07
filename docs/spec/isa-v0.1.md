# Chip Architecture v0.2

> 目标模型：Qwen2.5-0.5B（RMSNorm + SwiGLU + GQA + RoPE）。逐算子映射见
> docs/plans/ops-audit.md。v0.1 的指令流在 v0.2 下全部合法，opcode/kind 只追加不重排。

## 异构
    暂时不考虑添加RISC-V CPU

## 内存模型
    SRAM 0x10000000 - 0x10080000 共512KB
    DDR 0x80000000 - 0x100000000 共2G
        0x80000000 - 0x8000FFFF  命令队列区（64KB）：指令流 + descriptor，host 写入，NPU 取指
        0x80010000 - 0xFFFFFFFF  数据区：tensor 数据
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

5. ELEMENTWISE_ADD / ELEMENTWISE_MUL
    dst, lhs, rhs output/input1/input2 addr
    n 长度，元素数
    broadcast：允许一侧为可广播形状（如 [seq,1] 或 [1,N]），按行/列广播到另一侧。
        scale（RMSNorm gamma）、cos/sin（RoPE）、softmax 的减去 max 都靠它。

6. ACT
    dst, src output/input addr
    n 长度，元素数
    kind 元素级函数（ActKind 枚举）：
        RELU, EXP, RSQRT, RECIP, SILU
    RELU/SILU 是激活；EXP/RSQRT/RECIP 是特殊函数（softmax/RMSNorm 的归一化数学），
    只是指令形状相同，一道按 kind 区分。kind 只往后追加，RELU=0 不变。
    SILU(x) = x * sigmoid(x)。RSQRT = 1/sqrt(x)。RECIP = 1/x。

7. REDUCE
    dst, src addr
    n 长度，元素数
    kind 归约种类（ReduceKind 枚举）：MAX, SUM, SQUARE_SUM, ARGMAX
    axis 归约方向（ReduceAxis 枚举）：rowwise（沿最后一维，输出 [seq,1]），full（整块归约）
        rowwise 是 softmax/RMSNorm 的刚需（softmax 稳定与分母、RMSNorm 的 mean(x^2)）。
    ARGMAX 用于 lm_head 取每行最大位置；做流式分块合并时由编译器拼装。

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
        uint32_t n; // 元素数
        // broadcast：lhs/rhs 允许为 [seq,1] 或 [1,N]
    }

    struct ActParam {
        uint32_t dstAddr;
        uint32_t srcAddr;
        uint32_t n; // 元素数
        ActKind kind; // v0.2: RELU/EXP/RSQRT/RECIP/SILU
        union { //给其他激活传参数用
            uint32_t extra[4];
        }
    }

    struct ReduceParam {
        uint32_t dstAddr;
        uint32_t srcAddr;
        uint32_t n; // 元素数
        ReduceKind kind;
        ReduceAxis axis;
        union {
            uint32_t extra[4];
        }
    }

    // 从15x32的tensor切出来15x31
    struct DMAParam loadMatmulLhs{
        sramAddr = 0x10000000;
        ddrAddr = 0x80010000;
        rows = 15;
        cols = 31;
        srcStride = 32*2;
        dstStride = 31*2;
    }

    struct DMAParam loadMatmulRhs{
        sramAddr = 0x100003B0; //loadMatmulLhs.sramAddr+15*31*2，然后再16字节对齐
        ddrAddr = 0x800103C0;
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
        ddrAddr = 0x80011340;
        rows = 15;
        cols = 63;
        srcStride = 63*2;
        dstStride = 63*2;
    }

    struct EwiseParam elementwiseAddParam{
        dstAddr = 0x100021E0;
        rhsAddr = matmulParam.dstAddr;
        lhsAddr = loadElementwiseRhs.sramAddr;
        n = 15*63;
    }

    struct ActParam actParam {
        dstAddr = 0x10002950;
        srcAddr = elementwiseAddParam.dstAddr;
        n = 15*63;
        kind = Relu(0);
    }

    struct DMAParam storeActDst{
        sramAddr = actParam.dstAddr;
        ddrAddr = 0x80011AC0;
        rows = 15;
        cols = 63;
        srcStride = 63*2;
        dstStride = 63*2;
    }

    // matmul+elementwise_add+relu
    // [15*31]*[31*63] = [15*63] -> [15*63] + [15*63] = [15*63] -> Relu([15*63]) = [15*63]
    // v0.2 不排 pipeline
    // 指令流放在命令队列区（0x80000000起），定长8字节，descriptor 也在命令队列区内
    0x80000000: DMA_LOAD  loadMatmulLhs
    0x80000008: DMA_LOAD  loadMatmulRhs
    0x80000010: SYNC      // 保证两个DMA_LOAD完毕
    0x80000018: MATMUL    matmulParam
    0x80000020: SYNC      // 保证MATMUL计算完毕
    0x80000028: DMA_LOAD  loadElementwiseRhs
    0x80000030: SYNC      // 保证bias搬运完毕
    0x80000038: ELEMENTWISE_ADD elementwiseAddParam
    0x80000040: SYNC      // 保证ELEMENTWISE_ADD计算完毕
    0x80000048: ACT       actParam
    0x80000050: SYNC      // 保证ACT计算完毕
    0x80000058: DMA_STORE storeActDst
    0x80000060: SYNC      // 保证DMA_STORE完毕

## cycle模型
    当前 cycle 模型只包含 DMA 突发、MAC 吞吐、SIMD 吞吐；bank 冲突、多端口并行、惩罚周期等微架构细节留到后续性能模型
    新指令估算：ELEMENTWISE_MUL / ACT / REDUCE 走 SIMD 引擎，= ceil(n / ELEM_PER_CYCLE)；
    MATMUL 转置与不转置开销一致；REDUCE 的 ARGMAX 额外计入少量合并开销（后续定）。

## v0.2 不做 / 推迟到 v0.3

- fp32 累加区 ACC + MOVER：down_proj 跨块 fp16 累加精度已达标（err=6.9e-4），推迟到 v0.3。
- DMA_LOAD_ASYNC + WAIT：opcode 编号已冻结，编译器调度不实现，留 v0.3 做双缓冲。
- 间接寻址：token 依赖地址（嵌入行、KV cache 位置、RoPE cos/sin 偏移）全部 per-input
  静态编译、烘焙进 DMA descriptor；间接 DMA 寻址留 v0.3。
- 硬件 padding 到 MAC 倍数：依赖 roofline 实验数据决策。
- uint8/int8 量化：fp16 全链路对拍通过之后再上。
- MATMUL 的 inplace/多指令融合：后续版本。
