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

    mlir-opt --one-shot-bufferize input.mlir
    mlir-opt --one-shot-bufferize="bufferize-function-boundaries" input.mlir
    module {
        func.func @matmul(%arg0: tensor<16x16xf16>, %arg1: tensor<16x16xf16>, %arg2: tensor<16x16xf16>) -> tensor<16x16xf16> {
            %0 = bufferization.to_buffer %arg1 : tensor<16x16xf16> to memref<16x16xf16, strided<[?, ?], offset: ?>>
            %1 = bufferization.to_buffer %arg0 : tensor<16x16xf16> to memref<16x16xf16, strided<[?, ?], offset: ?>>
            %2 = bufferization.to_buffer %arg2 : tensor<16x16xf16> to memref<16x16xf16, strided<[?, ?], offset: ?>>
            %alloc = memref.alloc() {alignment = 64 : i64} : memref<16x16xf16>
            memref.copy %2, %alloc : memref<16x16xf16, strided<[?, ?], offset: ?>> to memref<16x16xf16>
            linalg.matmul ins(%1, %0 : memref<16x16xf16, strided<[?, ?], offset: ?>>, memref<16x16xf16, strided<[?, ?], offset: ?>>) outs(%alloc : memref<16x16xf16>)
            %3 = bufferization.to_tensor %alloc : memref<16x16xf16> to tensor<16x16xf16>
            return %3 : tensor<16x16xf16>
        }
    }

    memref.copy %2, %alloc，这里是把dst复制到alloc的内存，memref.copy A,B。其实就是把A复制到B
    是否加这个参数，不只是 取决于 pass,也 取决于 op,如果 前后的 op是 要 tensor类型，那就不能用 这个 

9. --one-shot-bufferize PASS
    针对tensorin类型的dialect,把数学上tensor,转成代码领域的memref,然后尽量进行内存复用
    bufferize-function-boundaries选项：
        如果 不加 bufferize-function-boundaries 选项，说明 pass前后接口还是 在 tensor领域，中间包着memref，下面的pass需要知道内存分配信息。而加上，就是彻底的 转换了，前后的 psss,必须使用memref. 

    %alloc = memref.alloc() {alignment = 64 : i64} : memref<16x16xf16>
    表示地址64bit对齐

10. pass作用等级
    def EclipseAllocate : Pass<"eclipse-allocate", "::mlir::ModuleOp"> 
    说明作用在Module。在pass实际实现中，ModuleOp module = getOperation();

11. walk
    func->walk([&](memref::AllocOp alloc){});
    walk中是lambda,这里就是找到每个一个allocop的节点，执行函数体内的动作

12. mlir IR结构
    module,func.func,op都是op
    block和region用来存放op的容器，针对下面分支的场景。scf.if就有了两个region,每个region有一个block，光有block无法满足分支结构，因为一个 Operation 不能直接拥有多个 Block，但它可以拥有多个 Region。

    scf.if %cond {     // <--- scf.if 是一个 Op
        // scf.if 拥有 2 个 Region（"then" 和 "else" 各一个）
        // then Region 里有 1 个 Block
        %0 = arith.addi ... 
    } else {
        // else Region 里有 1 个 Block
        %1 = arith.subi ...
    }

    区分region
    IR中{}内的一定是一个 region,带^bb的 一定是 block,但是 一个 {}内只有一个 block的情况，那就{}里面既是 region也是block，block被省略

13. builder和rewriter
    OpBuilder 可以在任意IR位置创建op
    PatternRewriter 在RewritePattern中必须使用rewriter创建和替换op,方便RewritePattern递归标准化/Lowering
    matchaddrewrite 在RewritePattern要重写的函数，代表看到某个结构替换
    
    创建完了之后，也应该 从 之前的 value出发 ，进行 替换 和删除

14. getZExtValue
    0扩展，从apint转为uint64_t,还有getSExtValue,带符号扩展，返回值是int64_t

15. funcOp.getNumResults()和funcOp->getNumResults()
    funcOp->getNumResults()，因为 funcOp也是 op,所以 返回的 是 那个 %value,当做了 普通 op去调用。而 funcOp.getNumResults()，才是 真的 去 获取return的 value数量
    