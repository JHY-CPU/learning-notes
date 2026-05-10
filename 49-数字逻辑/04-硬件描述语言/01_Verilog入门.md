# Verilog入门


## 一、模块定义（Module）


Verilog程序的基本单元是模块（module），每个模块描述一个电路功能。


### 1.1 模块基本结构


```
// 模块定义的基本格式
module 模块名(
    input  端口列表,
    output 端口列表,
    inout  双向端口列表
);
    // 内部信号声明
    wire  线网型信号;
    reg   寄存器型信号;

    // 功能描述
    ...

endmodule
```


### 1.2 示例：与门模块


```
// 2输入与门
module and_gate(
    input  a,
    input  b,
    output y
);
    assign y = a & b;  // 连续赋值语句
endmodule
```


## 二、数据类型


### 2.1 线网型（Net）


| 类型 | 说明 | 默认值 |
| --- | --- | --- |
| wire | 单根连线 | 高阻 z |
| tri | 三态线（多驱动源） | 高阻 z |


### 2.2 寄存器型（Reg）


- `reg`
   ：存储数据，不一定是硬件寄存器
- 在 always 块中赋值的信号必须声明为 reg
- 默认值为 x（未知）


### 2.3 参数定义


```
// 参数常量
parameter WIDTH = 8;
parameter IDLE = 2'b00, RUN = 2'b01, STOP = 2'b10;

// 数值表示
8'b1010_0101   // 8位二进制
8'd165         // 8位十进制
8'hA5          // 8位十六进制
8'o245         // 8位八进制
```


## 三、运算符


| 类别 | 运算符 | 说明 | 优先级 |
| --- | --- | --- | --- |
| 算术 | + - * / % | 加减乘除取模 | 高 |
| 位运算 | ~ & \| ^ ^~ | 按位非与或异或同或 | ↓ |
| 逻辑 | ! && \|\| | 逻辑非与或 | ↓ |
| 关系 | > < >= <= | 大小比较 | ↓ |
| 等式 | == != === !== | 等于/不等于/case等式 | ↓ |
| 移位 | << >> | 左移/右移 | ↓ |
| 缩减 | & ~& \| ~\| ^ ^~ | 单目运算，对向量逐位 | 低 |


> **Note:** **区分 === 和 ==：**
>
>
> `==`
> 比较时，x和z视为"不确定"，结果可能是x
>
>
> `===`
> 严格比较，x和z作为确定值参与比较（testbench常用）


## 四、赋值语句


### 4.1 连续赋值（assign）


```
// 组合逻辑，右值变化立即更新左值
assign y = a & b;         // 与门
assign {cout, sum} = a + b + cin;  // 全加器
```


### 4.2 过程赋值


```
// 阻塞赋值 = （组合逻辑用）
always @(*) begin
    temp = a & b;
    y = temp | c;
end

// 非阻塞赋值 <= （时序逻辑用）
always @(posedge clk) begin
    q1 <= d;
    q2 <= q1;  // q1和q2同时更新
end
```


> **Important:** **黄金法则：**
>
>
> 组合逻辑 always 块用阻塞赋值（=）
>
>
> 时序逻辑 always 块用非阻塞赋值（<=）
>
>
> 永远不要在同一个 always 块中混用两种赋值！


## 五、always块


### 5.1 组合逻辑 always 块


```
// 敏感列表用 *，表示所有输入信号
always @(*) begin
    case(sel)
        2'b00: y = a;
        2'b01: y = b;
        2'b10: y = c;
        default: y = d;
    endcase
end
```


### 5.2 时序逻辑 always 块


```
// 上升沿触发的D触发器
always @(posedge clk) begin
    if (rst)
        q <= 1'b0;
    else
        q <= d;
end

// 下降沿触发
always @(negedge clk) begin
    ...
end
```


## 六、常用电路描述


### 6.1 多路选择器


```
// 4选1 MUX
module mux4to1(
    input  [1:0] sel,
    input  [3:0] d,
    output reg y
);
    always @(*) begin
        case(sel)
            2'd0: y = d[0];
            2'd1: y = d[1];
            2'd2: y = d[2];
            2'd3: y = d[3];
        endcase
    end
endmodule
```


### 6.2 译码器


```
// 3-8译码器
module decoder3to8(
    input  [2:0] in,
    input       en,
    output reg [7:0] out
);
    always @(*) begin
        if (en)
            out = 8'b0000_0001 << in;
        else
            out = 8'b0000_0000;
    end
endmodule
```


### 6.3 计数器


```
// 模10计数器
module counter_mod10(
    input       clk,
    input       rst,
    output reg [3:0] count
);
    always @(posedge clk) begin
        if (rst)
            count <= 4'd0;
        else if (count == 4'd9)
            count <= 4'd0;
        else
            count <= count + 1;
    end
endmodule
```


### 6.4 移位寄存器


```
// 4位右移寄存器
module shift_reg(
    input       clk,
    input       din,    // 串行输入
    output reg [3:0] q
);
    always @(posedge clk) begin
        q <= {din, q[3:1]};  // 右移，高位补din
    end
endmodule
```


## 七、Testbench编写


Testbench是用于仿真的测试模块，不需要输入输出端口。


```
// 测试与门模块
module tb_and_gate;
    reg  a, b;
    wire y;

    // 实例化被测模块
    and_gate uut(
        .a(a),
        .b(b),
        .y(y)
    );

    // 产生激励信号
    initial begin
        $dumpfile("wave.vcd");   // 波形输出文件
        $dumpvars(0, tb_and_gate);

        a = 0; b = 0; #10;
        a = 0; b = 1; #10;
        a = 1; b = 0; #10;
        a = 1; b = 1; #10;
        $finish;
    end

    // 监视输出
    initial begin
        $monitor("time=%0t a=%b b=%b y=%b", $time, a, b, y);
    end
endmodule
```


> **Note:** **常用系统任务：**
>
>
> `$monitor`
> ：持续监视信号变化并打印
>
>
> `$display`
> ：一次性打印
>
>
> `#10`
> ：延迟10个时间单位
>
>
> `$finish`
> ：结束仿真
>
>
> `$dumpfile / $dumpvars`
> ：生成波形文件


## 八、知识要点总结


1. Module是Verilog的基本单元，描述电路模块
2. wire表示连线，reg表示存储（always块中赋值用reg）
3. assign用于组合逻辑的连续赋值
4. always @(*) 描述组合逻辑，always @(posedge clk) 描述时序逻辑
5. 组合逻辑用阻塞赋值=，时序逻辑用非阻塞赋值<=
6. case语句对应多路选择器，if-else对应优先逻辑


<!-- Converted from: 01_Verilog入门.html -->
