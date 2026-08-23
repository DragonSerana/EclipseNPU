1. 工具链切分和算子库切分的区别
    工具链能知道算子的前后关系，因此可实现更高级的切分，比如conv+matmul。而算子库只能看到一个算子，针对这个算子做详细的tile.算子库做"深"，工具链做"广"。工具链要面对的是任意组合。

2. cmodel和simulator
    cmodel是对芯片的模拟，simulator是装着芯片的整个系统，持有cmodel，x86操作芯片的接口。

3. memref
    全称memory reference，类似指针，不过还携带着 shape/布局/类型的信息
    eclipse.dma_load %a, %aView
    : memref<128x16xf16, 0>, memref<128x16xf16, strided<[128,1]>, 1>
    %a目标地址，%aView原地址，memref<128x16xf16, 0>，目标地址在SRAM，shape是128x16，type是f16。
    memref<128x16xf16, strided<[128,1]>, 1>，ddr多了一个stride信息，代表h到h+1=128，w到w+1=1

4. type.getLayout().isIdentity();
    判断布局是否是恒等，memref定义是否函数stride

5. Affine Dialect
    它是用来把多维数组的索引（比如 [i, j]），翻译成内存的一维物理地址的数学公式。
    比如每行有16个元素，那[i,j]->(i*16+j)
    把tensor中每个元素的位置具象到内存地址怎么算

6. 行主序
    strided<[128, 1]>是 行主序。H的 stride是 128。C语言默认的多维数组存储方式，先排行，再排列

7. ptional<ArrayRef<int64_t>>
    auto strides = type.getStrides();
    if (!strides.has_value())
        return false;
    ArrayRef<int64_t> strideVals = *strides;

    getStrides返回Optional<ArrayRef<int64_t>>，*strides就是打开 Optional这个封装.

8. linalg dialect
    LINEar ALGebra，线性代数
    本质其实是数学上的矩阵乘法，这阶段可以进行数学上就成立的变换，硬件无关