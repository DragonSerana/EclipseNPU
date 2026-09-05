1. 类的写法
    名词作为变量，动词作为函数。写类之前确认这个类的功能。
    默认是private，对外的接口使用public,剩下的类自己用的使用private
    构造函数负责"对象刚出生时状态必须是合法的"——如果你的类存在"不初始化就没法用"的状态，就需要构造函数。
    构造函数要写在public,因为外部实例化要用 
    构造函数最好也在CPP实现

    uint64_t computeCycles(const Instruction &inst) const;
    最后的const修饰this指针，表示不会修改类的成员变量

2. std::deque
    双端队列，两头插入都很快

3. constexpr
    编译期常量，要求编译期计算完毕。但是比define多了类型检查
 
4. const
    const 永远修饰它左边最近的那个东西；如果左边没有东西，才修饰右边最近的那个。而且最多修饰一个。
    const DMAParam* desc，左边没东西，那就修饰右边，也就是DMAParam，而不是 DMAParam*
    DMAParam* const desc，左边有东西，*，那也就说指针是const
    DMAParam  const* desc，左边有东西，DMAParam，那也就说值是const

    uint64_t computeCycles(const Instruction &inst) const;
    最后的const作用于this指针，这是一个常量成员函数。该函数不会修改调用它的那个对象的任何成员变量

5. ceil
    返回大于或等于给定数字的最小整数，向上舍入

6. queue_取值
    先move再删除适用于大资源，堆指针防止深拷贝。对于纯数据不需要std::move
    Instruction inst = std::move(queue_.front());
    ->
    Instruction inst = queue_.front();
    
    queue_.pop_front();

7. std::error_code EC
    相当于errno
    
8. llvm文件读写
    raw_fd_ostream纯写入 raw_fd_stream随机读写
    llvm::raw_fd_ostream fileOS(outputFileName, EC, llvm::sys::fs::OF_None);
    sys::fs::OF_None: 直接覆盖 
    写入直接通过
        fileOS << llvm::formatv

9. 匿名namespace
    相当于本文件内部私有，不会与其他文件的同名函数发生链接错误。

10. 算法时间复杂度
    O1, unordermap,通过对key取hash，直接找到数组下标
    On, for循环一次
    Olog2n，二分/二叉树
    On^2, 两个for循环嵌套
    On!, 数组内所有等于数组长度的排列
    O2^n, 

11. unordermap
    通过hash+桶的方式
    比如一对key,value。对key做hash,然后取模(hash(key)%10)，这样可以把很多key:value放在桶中，如果重复，那就往后跟链表，这样上来先算hash,O1，然后再链表找key,最差是On.不过当size>bucket_count时，会扩容，让链表尽量短
    比如结构体指针数组的好处是，key必须是正整数，而且不能是跳着的，比如 key=10000，按就要申请10000个sizeof(p)