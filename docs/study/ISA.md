# ISA指令集
DMA_LOAD+CONV+DMA_STORE是三个指令，组成一个序列
命令队列是存放这些指令序列的容器。
想象一下，你不仅要算一个Conv，你要算一个ResNet50，里面有几十个Conv。
    编译器：生成了几十个这样的指令序列。
    运行时：把这些序列像“排队”一样，依次放入一个环形缓冲区。
    硬件：NPU的控制单元从这个队列里依次取出序列执行。

一段对应的汇编指令
// 编译器生成的代码（放在内存0x8000处）
0x8000: DMA_LOAD  DDR_addr1, SRAM_addr1, 1024 //load input
0x8004: DMA_LOAD  DDR_addr2, SRAM_addr2, 1024 //load weight
0x8008: CONV      SRAM_addr1, SRAM_addr2, SRAM_addr3, params //卷积的input/weight在SRAM的地址，addr3是放结果的。 
0x800C: DMA_STORE SRAM_addr3, DDR_addr3, 1024//output放回DRR
0x8010: END       // 标记这个任务包结束

0x8000、0x8004 这些是PC（程序计数器）值，也就是指令在内存中的地址。
DMA_LOAD 就是方便人类阅读的助记符

疑问： 
1. pamas放参数，这个参数是在哪里的？我编译器组指令的时候 ，放在给NPU的指令里面？
    不会，因为参数挺长，放在指令太浪费，所以一般会配置在寄存器，类似外设寄存器
    这叫CSR（Control Status Registers，控制状态寄存器）
2. 这个指令，比如DMA_LOAD/CONV/SET_REG执行主体是谁？
    这里指令的执行者是NPU,他有执行指令的能力，CPU把指令放好在DDR之后，然后通过驱动告诉NPU开始执行指令，后面NPU主动去0x8000取指令，做从DDR把数据搬到SRAM的操作。所以这段汇编的视角是NPU,CPU的驱动有自己的汇编，主要是启动NPU。
3. 0x8000、0x8004是指令在内存中的地址。这句话怎么理解？是ELF驱动代码的位置？0x8000是给CPU看的还是NPU看的？
    那这里就明白了，0x8000确实是内存中的地址，具体这个地址放的什么指令，是CPU干的，CPU把给NPU运行的二进制，放在0x8000这个地方，然后告诉NPU从这里开始执行，然后NPU自己的PC寄存器就指向了0x8000，然后开始顺序执行。NPU也有自己的PC指针。
4. 是不是DDR对CPU和NPU不是完整可见的？CPU和NPU都只能看到自己的那部分。所以异构需要Device Copy.
    是完整可见，CPU看的是虚拟地址，而NPU是物理地址，所以需要映射下，把虚拟地址改成物理地址，或者物理改成虚拟，这样，另外一个处理单元就能接着往下计算了

// 第一步：配置卷积参数
SET_REG  REG_CONV_PAD,      1      // 配置pad为1
SET_REG  REG_CONV_STRIDE,   1      // 配置stride为1
SET_REG  REG_CONV_KERNEL_H, 3      // 配置kernel高为3
SET_REG  REG_CONV_KERNEL_W, 3      // 配置kernel宽为3

// 第二步：执行计算（此时CONV指令只需要知道地址）
CONV     SRAM_addr1, SRAM_addr2, SRAM_addr3

# NPU核心 
    NPU核心与CPU不同，一个NPU Engine，内部有很多MAC(Multiply-Accumulate)单元，比如，256甚至更多,一次可以执行256次MAC运算

# ISA设计
    SYNC指令即插fence，conv/dma_loadi只是把指令下发，SYNC才是等待数据执行完毕
    通过流水线可以隐藏DMA_LOAD/STORE的时间，但是由于木桶效应，读取效率仍然受读取时间限制，所以可以放大tile,尽量喂饱计算单元
    一个 MAC 阵列，比如 16×16，16×16 = 256 个单元，1 个周期 = 256 次乘法 + 256 次加法（统称为 256 次 MAC 操作）。

# 矩阵乘法
    [1,2,3]
    [4,5,6]
    *
    [11,12]
    [13,14]
    [15,16]
    我理解的 矩阵乘法  就是 
    [1*11+2*13+3*15,1*12+2*14+3*16]
    [4*11+5*13+6*15, 4*12+5*14,6*16]
    并行其实是下面这样并行，所以其实最大能利用MAC数，就是output shape,也就是M*N，规约轴K代表累加要跑多少次 
    MAC0:1*11+2*13+3*15
    MAC1:1*12+2*14+3*16
    MAC2:4*11+5*13+6*15
    MAC3:4*12+5*14,6*16

# 数据类型
    数据类型：fp16 和 int8 起步（int8 留给量化阶段）。
    这里 不管是 FP16还是 int8,都是 同时最多256次乘法计算，就是浮点乘法要的晶体管多

# 指令 
    DMA_LOAD sram_addr, ddr_addr, rows, cols, src_stride
    DMA_STORE
        src_stride用来切小方块，比如我从1920*1080大方块要3rows，5cols的小方块(3行5列)，那就要配置src_stride就是1920，实际跳过了1920-5=1915，跳过这么多像素
        指令的字段，filed,可以理解成函数的参数，放在NPU的DMA寄存器。
        这个指令集有两种常见的方式，一个是指令里面只有DMA_LOAD，在DMA_LOAD之前把参数通过MOV指令挪到NPU DMA的CSR，还有就是带个Descriptor，放着包含这些参数的结构体的指针
    
    MATMUL	dst, lhs, rhs, M, N, K, accumulate
        accumulate = 0（覆盖模式）：dst = LHS × RHS
        accumulate = 1（累加模式）：dst = dst + (LHS × RHS)

        所以 这个 参数的 目的就是 ，是否加上 SRAM上 本来的那个 dst的 值，一个可以用来切分超大块，比如 MAC 16*16,而我的 矩阵是 1024*1024 ，一次 MAC一行 都放不下 ，需要在 累加器之外 ，在进行 一次累加。还有一个 可以用来 算加 Bias
    EWISE_ADD	dst, lhs, rhs, n
    ACT	dst, src, n, kind(RELU/GELU)
        n是element wize的长度，kind是激活的类型
    SYCN
        插fence,等待前面的指令执行成功
    
# 指令格式
    不做二进制编码，就用 C++ struct：

    // runtime/include/eclipse_isa.h
    enum class OpCode : uint32_t { DMA_LOAD, DMA_STORE, MATMUL, EWISE_ADD, ACT, SYNC };

    struct Instruction {
    OpCode opcode;
    uint32_t dst, srcA, srcB;      // SRAM/DDR 地址（统一地址空间的偏移）
    uint32_t M, N, K;              // 形状参数（EWISE 只用 N）
    uint32_t strideA, strideB;
    uint32_t flags;                // bit0: accumulate, bit1-3: act kind, ...
    };

# pitch和字节对齐
    pitch，行间距。是二维矩阵按行存放时，相邻两行起始地址之间的距离
    字节对齐，为了减少内存突发,让下一个读取内存操作的起始地址是16的倍数。内存一口气上来的16byte，地址必须是16的倍数。
    如果我要15个数，比如uint8 a[15]。内存控制器其实一口气拿了16个byte,然后只用其中的15个是吧。就是下次分配buffer的时候，是从15+1开始往后 申请地址，这样如果数据是b[2]，内存控制器就可以i一口气拿上来，不是先拿16里面的1byte ，再拿上里17的1byte数据。