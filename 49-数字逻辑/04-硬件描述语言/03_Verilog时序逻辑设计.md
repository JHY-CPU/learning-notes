# Verilog时序逻辑设计

## 时序逻辑基础

时序逻辑电路的输出不仅取决于当前输入，还取决于电路的当前状态（即历史信息）。时序逻辑需要时钟信号来同步状态的变化，所有状态更新都在时钟边沿触发。

在 Verilog 中，时序逻辑使用 `always @(posedge clk)` 或 `always @(negedge clk)` 描述。时序逻辑中的信号必须声明为 `reg` 类型，使用非阻塞赋值 `<=`。

```verilog
module d_flip_flop (
    input  wire clk,
    input  wire rst_n,    // 低电平异步复位
    input  wire d,
    output reg  q
);
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            q <= 1'b0;
        else
            q <= d;
    end
endmodule
```

**阻塞赋值与非阻塞赋值的区别**：
- **阻塞赋值（`=`）**：立即执行赋值，用于组合逻辑
- **非阻塞赋值（`<=`）**：在时钟边沿结束后同时更新所有信号，用于时序逻辑

在时序逻辑中必须使用非阻塞赋值，否则会导致仿真结果与实际硬件行为不一致。

## 寄存器与移位寄存器

寄存器是时序逻辑的基本存储单元，用于暂存数据。

```verilog
// 8位寄存器，带使能信号
module register_8bit (
    input  wire       clk,
    input  wire       rst_n,
    input  wire       en,
    input  wire [7:0] d,
    output reg  [7:0] q
);
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            q <= 8'h00;
        else if (en)
            q <= d;
    end
endmodule

// 4位移位寄存器（左移）
module shift_reg_4bit (
    input  wire       clk,
    input  wire       rst_n,
    input  wire       serial_in,
    output wire       serial_out,
    output wire [3:0] parallel_out
);
    reg [3:0] shift_reg;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            shift_reg <= 4'b0000;
        else
            shift_reg <= {shift_reg[2:0], serial_in};  // 左移，串入
    end

    assign serial_out   = shift_reg[3];        // 串行输出
    assign parallel_out = shift_reg;           // 并行输出
endmodule
```

## 计数器设计

计数器是数字系统中最常用的时序电路之一，用于计数时钟脉冲或产生定时信号。

```verilog
// 参数化模N计数器
module counter_modN #(
    parameter N      = 10,
    parameter WIDTH  = 4
) (
    input  wire             clk,
    input  wire             rst_n,
    input  wire             en,
    output reg  [WIDTH-1:0] count,
    output wire             overflow
);
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            count <= 0;
        else if (en) begin
            if (count == N - 1)
                count <= 0;
            else
                count <= count + 1;
        end
    end

    assign overflow = (count == N - 1) && en;
endmodule

// 可逆计数器（加减计数器）
module up_down_counter (
    input  wire       clk,
    input  wire       rst_n,
    input  wire       en,
    input  wire       dir,     // 1=加计数, 0=减计数
    output reg  [7:0] count,
    output wire       max,
    output wire       min
);
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            count <= 8'h00;
        else if (en) begin
            if (dir)
                count <= count + 1;
            else
                count <= count - 1;
        end
    end

    assign max = (count == 8'hFF);
    assign min = (count == 8'h00);
endmodule
```

## 阻塞与非阻塞赋值的陷阱

初学者最常犯的错误是在时序逻辑中使用阻塞赋值。以下示例展示了两种写法的区别：

```verilog
// 错误写法：使用阻塞赋值
always @(posedge clk) begin
    a = b;    // 立即赋值
    b = a;    // 此时 a 已经是新的值，实现的是 a -> a -> b 的效果
end

// 正确写法：使用非阻塞赋值
always @(posedge clk) begin
    a <= b;   // 同时更新
    b <= a;   // 此时 a 还是旧值，实现的是交换效果
end
```

**设计原则总结**：
1. 组合逻辑使用 `always @(*)` + 阻塞赋值 `=`
2. 时序逻辑使用 `always @(posedge clk)` + 非阻塞赋值 `<=`
3. 不要在同一个 `always` 块中混合使用两种赋值方式
4. 不要对同一个信号在多个 `always` 块中赋值
