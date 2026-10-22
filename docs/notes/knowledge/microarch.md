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

# 矩阵乘法与并行
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

# SIMD 引擎：一行归约怎么算

    128 条 lane 就是 128 个 ALU，一拍能同时处理 128 个数据（每条 lane 一个）。
    128 这个数就是 ELEM_PER_CYCLE，ELEMENTWISE 和 REDUCE 共用。

    ALU 是两个输入，但另一个输入是它自己寄存器里的旧值，所以一拍只进 128 个新数据，
    不是 256 个。

    以 896 个元素的一行打比方：
        第 1 拍： 0~127   进入 128 条 lane（只是装进寄存器，不比较）
        第 2 拍： 128~255 进入 128 条 lane，和寄存器里的值比大小
        ...
        第 7 拍： 768~895
        7 拍之后，lane j 手里是 max(e[j], e[j+128], ..., e[j+768])
    然后树形规约：128 -> 64 -> ... -> 1，再走 7 拍，得到整行的最大值。
    （索引跟着值一起走，所以 ARGMAX 顺便就拿到位置了。）

    所以 cols=896 的一行 = 7 拍读数 + 7 拍树 = 14 拍。

    一行长度不是 128 的整数倍时（比如 100），最后一拍多出来的 lane 是越界的，
    要保持不动（取最大填 -inf，求和填 0），不然会读到下一行的数据。

# 硬件引擎
MAC阵列     MATMUL                        16×16 = 256 MAC/拍
SIMD(ALU)  ELEMENTWISE / REDUCE / RELU   128 元素/拍
SFU        EXP / RSQRT / SILU            32 元素/拍
DMA        LOAD / STORE                  32 B/拍

# output-stationary 
    当输入大于MAC（16x16）阵列时，需要驻留。比如A不动，然后 B往后 挪 n/16次

# 带宽
①	SRAM → PE 阵列	SRAM 的读端口	取决于 bank 数
②	PE → SRAM（写回）	SRAM 的写端口	取决于 bank 数
③	DDR → SRAM	DMA 引擎	16 个数/拍（DMA_BYTES_PER_CYCLE=32B）

# SRAM bank
    SRAM的bank,是为了提高从SRAM到PE的并行度
    一个 cycle 最多搬运就是 bank 个数据，包括 sram到 pe或者 dma在 ddr和 sram来回搬运
    并不是按照顺序分，而是交叉分，比如0/4是一个bank,1/5是一个bank
    b0, b1, b2, b3, b0, b1, b2, b3
    bank是 32的原因 是 16x16的 mac阵列，一拍需要 A16个 ，B16个，一共 32个元素.一个 bank是 16bit,正好 放  fp16的 数据

# SRAM 
    1W1R模型,如果读写的地址和数据复用 ，然后通过另外的 pin区分 是读还是写 。不过这样的话，就做不到同时读写了。所以这里用两套。
    DMA 往 SRAM 写的时候，不消耗阵列的读带宽。

    // 组合读，这样是不不合理，因实际物理需要晶体管变动，所以一般是同步读，即clk到来后再给出
    assign rdata = mem[raddr]; 

# DDR
    DDR 这个名字就是 Double Data Rate：数据在时钟的上升沿和下降沿各传一次
    x16指的是 16根线，一次 只能传递 16bit,2Byte的 数据，有因为 一拍 上升沿和下降沿都能传递数据，所以 是 4byte，一次 突发 16byte，需要 4 cycle
    clk_ddr和 clk_npu是不一样的，clk_ddr一个沿 是2个byte,一个 cycle是 4byte,然后 clk_npu侧 ，一拍 可以 收 32byte,对应 两次 突发

# 突发
    这个 突发，只是个 传输 单位，打包方式，不是 速率，所以 和 clk没有任何关系，这里 突发 默认 是 16byte,如果 一个 周期 穿 32 byte数据，那就是 两次突发，如果 一个周期 48byte个 数据， 那就是 三次 突发

    DDR获取 地址  基本 单位 就是 突发，比如 0x0e开始的 16byte数据，要 拿到 突发 0,和 突发 1 的 数据，然后  拼起来
    BL（ Burst Length）8，16-bit 总线 burst_bytes = BL × DDR数据总线位宽 / 8

    DMA_BURST_BYTES     = 16 字节   ← 来自【内存接口】
                                   x16 引脚 × BL8 = 16 字节，内存读取最小单位
                                   为了平摊启动成本的固定延迟，所以添加了内存读取最小单位。
    DMA_BYTES_PER_CYCLE = 32 字节   ← 来自【MAC 阵列】
                                    阵列一行操作数 = 16 个 fp16 = 32 字节，每cycle能读取的Byte数
                                    DMA_BYTES_PER_CYCLE是因为 我们 MAC要吃 16个 fp16，是硬件的设计 方向 ，不是因为 硬件 得到 这个数，而是基于 我们的 NPU要 16个 FP16的 MAC,硬件 要尽可能的 保证 cycle是 32
    比如说 输入MN是 128x128,MAC是 16x16,我一次从SRAM获取的数据要32(决定 SRAM 读端口数 = 32 个 bank)*2Byte,因为 我每行 数据 要复用 8次，所以 dmaload的速度，最好 是 64/8=8Byte数据。如果 输入 是 16x16,MAC没有 一点 复用，所以 dmaload的速度最好是64Byte.这里我们 选择 32 ，是为了兼顾两种 情况 ，而且不 阻碍效率，也不浪费面积，硬件要满足这个设计，通过给DDR_CLK提高频率等方式，在突发是16byte的基础上，每NPU_CLK吞吐率Byte数据，满足我的MAC要求。

    默认 damload ,ddr那边1cycle可以提供32Byte的数据，至于 物理上 怎么 实现，我们 不管 。一般 是 通过 多通道/多 bank实现的一次读取两个最小单位。实际其实突发和DDR给数据的速率也不是两倍大小。
    DDR 读取数据的时钟，不和我从 sram这侧 看到 ddr给 数据的 时钟(dmaload)相同 ，到 sram这里，已经是 NPU_CLK了，这是突发和ddr提供数据每cycle不同最主要的 原因