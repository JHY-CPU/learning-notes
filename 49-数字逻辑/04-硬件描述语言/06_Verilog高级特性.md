# Verilog高级特性

## Generate 块

`generate` 块是 Verilog-2001 引入的特性，允许在编译时（elaboration time）根据条件或循环生成硬件结构。它使得参数化设计和重复结构的描述更加简洁。

`generate` 块支持三种结构：`generate for`、`generate if` 和 `generate case`。

```verilog
// 使用 generate for 实现参数化位宽的多路选择器
module mux_generate #(
    parameter WIDTH = 8,
    parameter SEL_BITS = 3
) (
    input  wire [(2**SEL_BITS)*WIDTH-1:0] din_flat,
    input  wire [SEL_BITS-1:0]            sel,
    output wire [WIDTH-1:0]               dout
);
    wire [WIDTH-1:0] din_array [0:(2**SEL_BITS)-1];

    // 将扁平总线拆分为数组
    genvar i;
    generate
        for (i = 0; i < 2**SEL_BITS; i = i + 1) begin : gen_unflatten
            assign din_array[i] = din_flat[(i+1)*WIDTH-1 : i*WIDTH];
        end
    endgenerate

    assign dout = din_array[sel];
endmodule
```

`generate` 块的另一个常见用途是根据参数条件选择不同的实现：

```verilog
// 根据参数选择不同的加法器实现
module adder_configurable #(
    parameter USE_CARRY = 0,
    parameter WIDTH     = 8
) (
    input  wire [WIDTH-1:0] a,
    input  wire [WIDTH-1:0] b,
    input  wire             cin,
    output wire [WIDTH-1:0] sum,
    output wire             cout
);
    generate
        if (USE_CARRY) begin : gen_carry_chain
            // 进位链加法器实现
            wire [WIDTH:0] carry;
            assign carry[0] = cin;

            genvar j;
            for (j = 0; j < WIDTH; j = j + 1) begin : gen_fa
                assign sum[j]    = a[j] ^ b[j] ^ carry[j];
                assign carry[j+1] = (a[j] & b[j]) | (a[j] & carry[j]) | (b[j] & carry[j]);
            end
            assign cout = carry[WIDTH];
        end else begin : gen_simple
            // 简单加法器实现
            wire [WIDTH:0] result;
            assign result = a + b + cin;
            assign sum    = result[WIDTH-1:0];
            assign cout   = result[WIDTH];
        end
    endgenerate
endmodule
```

## Function 与 Task

`function` 和 `task` 是 Verilog 中用于代码复用的两种结构。

### Function

函数必须在零仿真时间内完成，不能包含延迟语句（`#`）、`always` 或 `initial` 块。函数至少有一个输入参数，返回一个值。

```verilog
module func_example (
    input  wire [7:0] a,
    input  wire [7:0] b,
    output wire [7:0] gcd_result
);
    // 计算最大公约数的函数
    function [7:0] gcd;
        input [7:0] x, y;
        reg [7:0] temp;
        begin
            while (y != 0) begin
                temp = y;
                y    = x % y;
                x    = temp;
            end
            gcd = x;
        end
    endfunction

    assign gcd_result = gcd(a, b);

    // 奇偶校验函数示例
    function parity;
        input [7:0] data;
        begin
            parity = ^data;  // 归约异或，所有位异或
        end
    endfunction
endmodule
```

### Task

任务比函数更灵活，可以包含时序控制（延迟、`@`、`wait`），可以有多个输入输出参数，但不能返回值。

```verilog
module task_example (
    input  wire       clk,
    input  wire       rst_n,
    input  wire [7:0] addr,
    input  wire [7:0] wdata,
    input  wire       wr_en,
    output reg  [7:0] rdata
);
    reg [7:0] memory [0:255];

    // 写任务
    task mem_write;
        input [7:0] address;
        input [7:0] data;
        begin
            @(posedge clk);
            memory[address] <= data;
            $display("[%0t] MEM WRITE: addr=0x%h data=0x%h",
                     $time, address, data);
        end
    endtask

    // 读任务
    task mem_read;
        input  [7:0] address;
        output [7:0] data;
        begin
            @(posedge clk);
            data = memory[address];
            $display("[%0t] MEM READ: addr=0x%h data=0x%h",
                     $time, address, data);
        end
    endtask

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rdata <= 8'h00;
        end else if (wr_en) begin
            mem_write(addr, wdata);
        end else begin
            mem_read(addr, rdata);
        end
    end
endmodule
```

## 编译指令与预处理

Verilog 提供了一系列编译指令，以 `` ` `` 开头，在编译前进行预处理。

```verilog
// 常用编译指令
`timescale 1ns / 1ps    // 时间单位/精度
`define DATA_WIDTH 32    // 宏定义
`undef DATA_WIDTH        // 取消宏定义
`include "header.vh"     // 包含文件
`ifdef SIM_MODE          // 条件编译
    // 仿真专用代码
`else
    // 综合专用代码
`endif
`default_nettype none    // 禁止隐式网表声明（推荐）
`resetall                // 重置所有编译指令

// 使用宏定义的示例
`define ADD  3'b000
`define SUB  3'b001
`define AND  3'b010
`define OR   3'b011
`define XOR  3'b100

module alu_macro (
    input  wire [7:0] a,
    input  wire [7:0] b,
    input  wire [2:0] op,
    output reg  [7:0] result
);
    always @(*) begin
        case (op)
            `ADD: result = a + b;
            `SUB: result = a - b;
            `AND: result = a & b;
            `OR:  result = a | b;
            `XOR: result = a ^ b;
            default: result = 8'h00;
        endcase
    end
endmodule
```

`defparam` 可以在模块外部覆盖参数，但在现代综合工具中不推荐使用，建议使用 `#()` 参数覆盖语法。

```verilog
module counter_defparam #(
    parameter MAX_VAL = 99
) (
    input  wire clk,
    input  wire rst_n,
    output reg [7:0] count
);
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            count <= 0;
        else if (count >= MAX_VAL)
            count <= 0;
        else
            count <= count + 1;
    end
endmodule

// 不推荐的 defparam 写法
module top_bad;
    wire [7:0] cnt;
    counter_defparam u_cnt (.clk(clk), .rst_n(rst_n), .count(cnt));
    defparam u_cnt.MAX_VAL = 49;  // 不推荐
endmodule

// 推荐的参数覆盖写法
module top_good;
    wire [7:0] cnt;
    counter_defparam #(.MAX_VAL(49)) u_cnt (
        .clk(clk), .rst_n(rst_n), .count(cnt)
    );
endmodule
```

高级特性使用建议：
1. `generate` 块非常适合构建参数化的可复用模块
2. `function` 适合纯组合逻辑的计算（如校验、编码转换）
3. `task` 适合包含时序控制的操作（如总线读写）
4. 使用 `` `default_nettype none `` 强制显式声明所有信号类型，避免隐式声明导致的错误
