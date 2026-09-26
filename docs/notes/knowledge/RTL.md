1. RTL Register Transfer Level
编译命令 
    cd /home/serana/EclipseNPU
    verilator --lint-only -Wall --top-module counter rtl/pip/counter.sv

    verilator --cc --exe --build --trace -Wall --top-module counter \
    rtl/pip/counter.sv rtl/tb/pip/tb_counter.cpp -o tb_counter --Mdir build/rtl/pip

    ./build/rtl/pip/tb_counter
    gtkwave wave.vcd

2. 
verilator --lint-only -Wall --top-module counter rtl/pip/counter.sv

.sv 是 SystemVerilog 源代码文件的后缀
--lint-only 
    加上--lint-only之后，它不会生成 C++，也不会编译出可执行程序，只检查语法
    counter.sv  --->  Verilator 翻译成 C++  --->  编译成可执行程序  --->  跑仿真
--top-module counter
    module就是一小块电路，就是一个小的硬件功能。top-module是最外层模块


verilator --cc --exe --build --trace -Wall --top-module counter \
  rtl/pip/counter.sv tb/pip/tb_counter.cpp -o tb_counter --Mdir build/rtl/pip

--cc	把 SystemVerilog 代码翻译成 C++ 代码
--exe	最终要生成一个可执行程序，而不是只生成库
--build	翻译完以后，自动调用 make/g++ 编译链接
--trace	打开波形跟踪功能，让仿真器支持生成 VCD/FST 波形
tb/pip/tb_counter.cpp	C++ 测试平台，负责驱动时钟、复位、输入。如果 说 model是个 i2c芯片，tb/pip/tb_counter.cpp就 提供 vcc,gnd,clk，相当于给芯片(model) 能 跑起来的 周围电路

gtkwave wave.vcd 查看波形

3. systemverilog
module counter (
  input  logic       clk,
  input  logic       rst_n,
  input  logic       en,
  output logic [7:0] cnt
);
counter可以理解成一个芯片，input可以理解成 芯片的引脚输入，output是 并行 8根引脚输出。logic可以当成一根线或者一个信号

always_ff @(posedge clk) begin
    if (!rst_n)  cnt <= 8'd0;
    else if (en) cnt <= cnt + 8'd1;
end
assign cnt_next = cnt + 8'd1;

always_ff	always flip-flop	寄存器 / 时序逻辑(输出不仅看当前输入，还看过去存下来的状态，比如cnt)
posedge clk clk上升沿触发
8'd0 8位宽的，十进制0
<= ,clk到来之后，所有 <= 右边的表达式，用时钟沿之前的值（旧值）计算，所有 <= 左边的寄存器，在同一个时间点统一更新为新值
= ，立刻更新为新值，不需要等待clk

always_ff描述，有时钟，那就是寄存器，可以存值，比如clk。没时钟的就是导线，只有传输没有存值的功能

4. 寄存器
    寄存器，就是一组并排的D触发器。 D->D触发器->Q
   上升沿到来时，Q 采样的是“上升沿那一刻 D 的值”，然后在上升沿之后，Q 变成这个值。D触发器的作用就是，在时间轴上，把连续变化的 D，变成只在时钟边沿更新的 Q

    i实际写寄存器的时候是这样的 ，我 写了 寄存器 的 时候，0x01这个 数据 会放在 数据 总线，然后根据 写的 寄存器地址，给 对应的 model发 clk,然后 他就把 这个 0x01吃进去，然后 D触发器的 Q就把电平 值为1了

5. valid/data/ready

    上游     ──> s_valid   ┌────────────────┐  m_valid ──>  下游
            ──> s_data    │ 单级寄存器切片  │  m_data  ──>
            <── s_ready   └────────────────┘  <── m_ready

    valid（我有数据）和 data（数据）同向，由发送方驱动；
    ready（我能收）反向，由接收方驱动
    一次传输成立 = 同一个时钟沿上采到 valid && ready 都是 1。

6. 拍(cycle)和沿
    沿上 = 时钟跳变的那一瞬间，通常是上升沿 0 → 1
    一拍 = 一个时钟周期，也就是两个相邻上升沿之间的时间。
