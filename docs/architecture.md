# Chip Architecture v0.1

## 异构
    暂时不考虑添加RISC-V CPU

## 内存模型
    SRAM 0x10000000 - 0x10080000 共512KB
    DDR 0x80000000 - 0xc0000000 共1G
    字节对齐，tensor在内存上16字节对齐，编译器保证sram_addr/ddr_addr都是16字节对齐。

## 执行模型
    指令 = (opcode, desc_ptr)；desc_ptr 指向 DDR 中的参数结构体（descriptor）。

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
    src_stride 切小方块时，要跳过的数量，即原内存pitch。
        地址(i,j) = base + i*src_stride + j *dtype_size   (i in [0,rows), j in [0,cols))
    dst_stride 目标内存的pitch，packed 就是 dst_stride == cols * dtype_size，紧密排列

2. DMA_STORE
    同DMA_LOAD

3. MATMUL
    dst, lhs, rhs output/input1/input2 addr
    M, N, K MKN轴的长度，最大取值为1024
    accumulate accumulate = 0（覆盖模式），accumulate = 1（累加模式）

    编译器负责padding到MAC大小。
**TODO.后续支持硬件padding到MAC数量**    
    dst和src暂时不支持inplace，即输入输出共用一块内存
**TODO.后续支持inplace**
    内部 ACC是fp32累加，写回fp16

4. EWISE_ADD
    dst, lhs, rhs output/input1/input2 addr
    n 长度，字节，编译器根据dtype手动计算
**TODO.后续支持inplace**

5. ACT
    dst, src output/input addr
    n 长度，字节，编译器根据dtype手动计算
    kind 激活种类(RELU/GELU)
**TODO.后续支持inplace**

6. SYNC
    暂时无参数，表示fence all，等待所有指令执行完成

## 示例
**等上面的问题都解决完了，我再写一个示例**