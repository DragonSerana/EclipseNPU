# Chip Architecture v0.1

## 异构
    暂时不考虑添加RISC-V CPU

## 内存模型
    SRAM 0x10000000 - 0x10080000 共512KB
    DDR 0x80000000 - 0xc0000000 共1G
    字节对齐，tensor在内存上16字节对齐，编译器保证sram_addr/ddr_addr都是16字节对齐。
    内存控制器按照16Byte突发，如果地址不是16的倍数，需要突发两次，如果没有地址对齐，比如地址落在了0x0e的位置，如果后面一个数据有16byte，之前一次突发就能拿到，现在要两次
    计算指令（MATMUL/ELEMENTWISE/ACT）的操作数在 SRAM 中必须 packed（行连续）存储；只有 DMA 支持 stride。
    DDR中，input tensor是啥样就啥样，SRAM中必须是packed。

## 执行模型
    指令 = (opcode, desc_ptr)；desc_ptr 指向 DDR 中的参数结构体（descriptor）。

    编译器把descriptor放在DDR中，目前为了可读性通过struct表示。
    Host将指令放进循环队列，Simulator从队列读取code，读取后就删除。
    指令顺序发出，但是不保证上条指令执行完毕，如果有强依赖关系，插入SYNC指令等待所有Code执行完毕。

## 数据类型
    v0.1 仅支持fp16 
    fp16 dtype = 0, size = 16bit， 带符号
**TODO.后续支持uint8量化等**

## layout
    MATMUL v0.1 版本仅支持 MK*KN 
**TODO.后续再支持transpose**

## 字段单位及大小端
    字节寻址，即地址表示Byte，小端序。地址为32bit

## 指令集设计
1. DMA_LOAD
    sram_addr SRAM地址
    ddr_addr DDR地址，这里地址就是要搬运的地址，如果是整个input,那就是整个input的头，如果是小块，那就是工具链算好的小块的头
    rows 行(裁剪的小tensor行数)
    cols 列(裁剪的小tensor列数)
    src_stride 切小方块时，要跳过的长度，即原内存pitch。
        地址(i,j) = base + i*src_stride + j *dtype_size   (i in [0,rows), j in [0,cols))
    dst_stride 目标内存的pitch，packed 就是 dst_stride == cols * dtype_size，紧密排列

2. DMA_STORE
    同DMA_LOAD

3. MATMUL
    dst, lhs, rhs output/input1/input2 addr
    M, N, K MKN轴的长度，最大取值为1024
    accumulate accumulate = 0（覆盖模式），accumulate = 1（累加模式）

    cmodel 支持任意 M,N,K≤1024；16 对齐仅为性能；padding 是编译器的可选优化
**TODO.后续支持硬件padding到MAC数量**    
    dst和src暂时不支持inplace，即输入输出共用一块内存
**TODO.后续支持inplace**
    内部 ACC是fp32累加，写回fp16
    

4. ELEMENTWISE_ADD
    dst, lhs, rhs output/input1/input2 addr
    n 长度，字节，编译器根据dtype手动计算
**TODO.后续支持inplace**
**TODO.后续支持broadcast**

5. ACT
    dst, src output/input addr
    n 长度，字节，编译器根据dtype手动计算
    kind 激活种类(RELU/GELU)
**TODO.后续支持inplace**

6. SYNC
    暂时无参数，表示fence all，等待所有指令执行完成

## 示例
    struct dma_opcode_param { 
        uint32_t sram_addr;
        uint32_t ddr_addr; // 模拟空间的物理地址，不是x86的malloc地址
        uint32_t rows; //rows和cols都是元素数
        uint32_t cols;
        uint32_t src_stride;
        uint32_t dst_stride;
    }

    struct sync_opcode_param {
        // 目前是空，预留
    }    

    struct matmul_opcode_param { 
        // 除了DMA_LOAD/STORE，其他opcode的地址都是SRAM地址
        uint32_t dst_addr;
        uint32_t rhs_addr;
        uint32_t lhs_addr;
        uint32_t M;
        uint32_t K;
        uint32_t N;
        uint32_t accumulate;
    }

    struct elementwise_add_opcode_param { 
        uint32_t dst_addr;
        uint32_t rhs_addr;
        uint32_t lhs_addr;
        uint32_t n;
    }

    struct act_opcode_param {
        uint32_t dst_addr;
        uint32_t src_addr;
        uint32_t n;
        uint32_t kind;
        union { //给其他激活传参数用
            uint32_t extra[4];
        }
    }

    // 从15x32的tensor切出来15x31
    struct dma_opcode_param LOAD_MATMUL_LHS{
        sram_addr = 0x10000000;
        ddr_addr = 0x80000000;
        rows = 15; 
        cols = 31;
        src_stride = 32*2;
        dst_stride = 31*2;
    }

    struct dma_opcode_param LOAD_MATMUL_RHS{
        sram_addr = 0x100003B0; //LOAD_MATMUL_LHS.sram_addr+15*31*2，然后再16字节对齐
        ddr_addr = 0x800003C0; 
        rows = 31; 
        cols = 63;
        src_stride = 64*2;
        dst_stride = 63*2;
    }

    struct matmul_opcode_param MATMUL_PARMA{
        dst_addr = 0x10001300; // LOAD_MATMUL_RHS.sram_addr+31*63*2，然后再16字节对齐
        rhs_addr = LOAD_MATMUL_RHS.sram_addr;
        lhs_addr = LOAD_MATMUL_LHS.sram_addr;
        M = 15;
        K = 31;
        N = 63;
        accumulate = 0;
    }

    struct dma_opcode_param LOAD_ELEMENTWISE_ADD_RHS{
        sram_addr = 0x10001A70; // MATMUL_PARMA.dst_addr+15*63*2，然后再16字节对齐
        ddr_addr = 0x80001340;
        rows = 15; 
        cols = 63;
        src_stride = 63*2;
        dst_stride = 63*2;
    }

    struct elementwise_add_opcode_param ELEMENTWISE_ADD_PARMA{ 
        dst_addr = 0x100021E0; 
        rhs_addr = MATMUL_PARMA.dst_addr;
        lhs_addr = LOAD_ELEMENTWISE_ADD_RHS.sram_addr;
        n = 15*63*2;
    }

    struct act_opcode_param ACT_PARAM { 
        dst_addr = 0x10002950; 
        src_addr = ELEMENTWISE_ADD_PARMA.dst_addr;
        n = 15*63*2;
        kind = Relu(0);
    }

    struct dma_opcode_param STORE_ACT_DST{
        sram_addr = ACT_PARAM.dst_addr;
        ddr_addr = 0x80001AC0;
        rows = 15; 
        cols = 63;
        src_stride = 63*2;
        dst_stride = 63*2;
    }

    // matmul+elementwise_add+relu
    // [15*31]*[31*63] = [15*63] -> [15*63] + [15*63] = [15*63] -> Relu([15*63]) = [15*63]
    // v0.1不排pipeline
    0x8000: DMA_LOAD  LOAD_MATMUL_LHS
    0x8004: DMA_LOAD  LOAD_MATMUL_RHS
    0x8008: SYNC SYNC_PARAM // 保证两个DMA_LOAD完毕
    0x800C: MATMUL  MATMUL_PARMA
    0x8010: SYNC SYNC_PARAM // 保证MATMUL计算完毕        
    0x8014: DMA_LOAD LOAD_ELEMENTWISE_ADD_RHS
    0x8018: SYNC SYNC_PARAM // 保证DMA_LOAD完毕        
    0x801C: ELEMENTWISE_ADD ELEMENTWISE_ADD_PARMA
    0x8020: SYNC SYNC_PARAM // 保证ELEMENTWISE_ADD计算完毕  
    0x8024: ACT ACT_PARAM 
    0x8028: SYNC SYNC_PARAM // 保证ACT计算完毕  
    0x802C: DMA_STORE STORE_ACT_DST
    0x8030: SYNC SYNC_PARAM // 保证DMA_STORE完毕  