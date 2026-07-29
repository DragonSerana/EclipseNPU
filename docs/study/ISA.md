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