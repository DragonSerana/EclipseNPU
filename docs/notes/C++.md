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
    const DMAOpcodeParam* desc，左边没东西，那就修饰右边，也就是DMAOpcodeParam，而不是 DMAOpcodeParam*
    DMAOpcodeParam* const desc，左边有东西，*，那也就说指针是const
    DMAOpcodeParam  const* desc，左边有东西，DMAOpcodeParam，那也就说值是const