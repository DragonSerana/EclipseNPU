# Chip Architecture v0.1

## 异构
    暂时不考虑添加RISC-V CPU

## 内存模型
    SRAM 0x10000000 - 0x10080000 共512KB
    DDR 0x80000000 - 0xc0000000 共1G
        0x80000000 - 0x8000FFFF  命令队列区（64KB）：指令流 + descriptor，host 写入，NPU 取指
        0x80010000 - 0xBFFFFFFF  数据区：tensor 数据
    字节对齐，tensor在内存上16字节对齐，编译器保证sramAddr/ddrAddr都是16字节对齐。
    内存控制器按照16Byte突发，突发必须是16的倍数，如果地址不是16的倍数，需要突发两次，如果没有地址对齐，比如地址落在了0x0e的位置，如果后面一个数据有16byte，之前一次突发就能拿到，现在要两次
    计算指令（MATMUL/ELEMENTWISE/ACT）的操作数在 SRAM 中必须 packed（行连续）存储；只有 DMA 支持 stride。
    DDR中，input tensor是啥样就啥样，SRAM中必须是packed。

## 执行模型
    指令 = (opcode: u32, descPtr: u32)，定长 8 字节；descPtr 指向命令队列区中的参数结构体（descriptor）。
    SYNC 无参数，descPtr = 0。

    编译器把指令流和 descriptor 布局在命令队列区，目前为了可读性通过 struct 表示。
    Host将指令放进环形队列，Simulator从队列读取code，读取后就删除。
    指令顺序发出，但是不保证上条指令执行完毕，如果有强依赖关系，插入SYNC指令等待所有Code执行完毕。

## 数据类型
    v0.1 仅支持fp16 
    fp16 dtype = 0, size = 16bit， 带符号

## layout
    MATMUL v0.1 版本仅支持 MK*KN 

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

3. MATMUL
    dst, lhs, rhs output/input1/input2 addr
    M, N, K MKN轴的长度（字段上限1024，仅为位宽说明）
    accumulate: 0=覆盖模式，1=累加模式

    合法性约束：一次 MATMUL 的 dst+lhs+rhs 三块 tile 必须同时驻留 SRAM；
    违反时 simulator 直接 assert（真实硬件不检查，只会静默算错）。
    cmodel 支持任意 M,N,K≤1024；16 对齐仅为性能；padding 是编译器的可选优化
    dst和src暂时不支持inplace，即输入输出共用一块内存
    cmodel 禁止 dst/lhs/rhs 任意两块重叠（比硬件合法集保守；例如 A×A 合法但被拒），
    该保守行为在 dialect verifier 保持一致。

    数值语义：块内 fp32 累加，写回 fp16；accumulate=1 读回的 dst 是 fp16，
    跨 K-block 的累加误差由软件承担（已知行为，v0.2 引入 fp32 累加区解决）。
    

4. ELEMENTWISE_ADD
    dst, lhs, rhs output/input1/input2 addr
    n 长度，元素数

5. ACT
    dst, src output/input addr
    n 长度，元素数
    kind 激活种类（ActKind 枚举，v0.1 仅 RELU）

6. SYNC
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
    }

    struct EwiseAddParam { 
        uint32_t dstAddr;
        uint32_t rhsAddr;
        uint32_t lhsAddr;
        uint32_t n; // 元素数
    }

    struct ActParam {
        uint32_t dstAddr;
        uint32_t srcAddr;
        uint32_t n; // 元素数
        ActKind kind; // 激活种类，v0.1 仅 RELU
        union { //给其他激活传参数用
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
    }

    struct DMAParam loadElementwiseAddRhs{
        sramAddr = 0x10001A70; // matmulParam.dstAddr+15*63*2，然后再16字节对齐
        ddrAddr = 0x80011340;
        rows = 15; 
        cols = 63;
        srcStride = 63*2;
        dstStride = 63*2;
    }

    struct EwiseAddParam elementwiseAddParam{ 
        dstAddr = 0x100021E0; 
        rhsAddr = matmulParam.dstAddr;
        lhsAddr = loadElementwiseAddRhs.sramAddr;
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
    // v0.1不排pipeline
    // 指令流放在命令队列区（0x80000000起），定长8字节，descriptor 也在命令队列区内
    0x80000000: DMA_LOAD  loadMatmulLhs
    0x80000008: DMA_LOAD  loadMatmulRhs
    0x80000010: SYNC      // 保证两个DMA_LOAD完毕
    0x80000018: MATMUL    matmulParam
    0x80000020: SYNC      // 保证MATMUL计算完毕
    0x80000028: DMA_LOAD  loadElementwiseAddRhs
    0x80000030: SYNC      // 保证bias搬运完毕
    0x80000038: ELEMENTWISE_ADD elementwiseAddParam
    0x80000040: SYNC      // 保证ELEMENTWISE_ADD计算完毕
    0x80000048: ACT       actParam
    0x80000050: SYNC      // 保证ACT计算完毕
    0x80000058: DMA_STORE storeActDst
    0x80000060: SYNC      // 保证DMA_STORE完毕

## v0.2 规划（占位，待 H3 ops-audit 完成后冻结）

    目标模型：Qwen2.5-0.5B（RMSNorm + SwiGLU + GQA + RoPE；备选 TinyLlama 1.1B）。
    逐算子的指令缺口见 docs/ops-audit.md（H3 产出），本清单随审计结果修订。

    - REDUCE（max/sum/平方和）：Argmax、softmax、RMSNorm 前置
    - EWISE_MUL：SwiGLU（silu 结果逐元素乘）、RoPE 旋转的逐元素乘；v0.1 只有 ELEMENTWISE_ADD
    - DIV/RSQRT：RMSNorm（x/sqrt(mean(x^2)+eps)）、softmax 归一化
    - EXP / LUT：softmax、SiLU；若选 GELU 系模型则需 tanh/erf LUT
    - sin/cos LUT：RoPE
    - DMA_LOAD_ASYNC + WAIT tag：双缓冲，用计算掩盖搬运
    - fp32 累加区（ACC）：MAC 阵列出口的专用 fp32 RAM，独立地址空间（如 0x20000000，64KB），
      仅 MATMUL 和 MOVER 可访问；MATMUL 的 dst 指向 ACC 时 K 维接力全程 fp32，不再经过 SRAM 的 fp16；
      新增 MOVER 指令：ACC(fp32) → SRAM(fp16) 量化，可选融合 activation
    - 硬件 padding 到 MAC 倍数（依赖 roofline 实验数据决策）
    - uint8/int8 量化（后置：fp16 全链路对拍通过之后再上，避免精度问题污染集成问题）
    - EWISE/ACT inplace、broadcast（RMSNorm 乘 scale、RoPE 乘 scalar 等）
    - MATMUL transpose layout（KV cache / RoPE 场景）
    - 内存容量：DDR 扩到 2GB（fp16 权重约 1.4GB + 数据区；改宏即可）

    v0.2 仍无间接寻址：token 依赖地址（嵌入行、KV cache 位置）由 per-input 静态编译
    在编译期烘焙进 DMA descriptor；间接 DMA 寻址留 v0.3。
