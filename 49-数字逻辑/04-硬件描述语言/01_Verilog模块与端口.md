# Verilog模块与端口

## 模块定义基础

Verilog HDL 是硬件描述语言，模块（module）是其基本设计单元。每个模块描述一个独立的硬件功能块，包含端口声明、内部信号声明和功能实现。

模块的基本语法结构如下：

```verilog
module module_name (
    port_list
);
    // 端口声明
    // 内部信号声明
    // 功能实现
endmodule
```

一个简单的与门模块示例：

```verilog
module and_gate (
    input  wire a,
    input  wire b,
    output wire y
);
    assign y = a & b;
endmodule
```

模块的命名应具有描述性，遵循一定的命名规范。常见的命名风格包括：
- **小写下划线风格**：`alu_unit`、`fifo_controller`
- **驼峰命名风格**：`AluUnit`、`FifoController`

## 端口类型与方向

Verilog 中端口分为三种方向：

| 端口类型 | 关键字 | 说明 |
|---------|--------|------|
| 输入端口 | `input` | 信号从外部流入模块 |
| 输出端口 | `output` | 信号从模块流向外部 |
| 双向端口 | `inout` | 信号可双向传输 |

端口的数据类型可以是 `wire`、`reg` 或其他类型：
- `input` 端口默认为 `wire` 类型，不能声明为 `reg`
- `output` 端口可以是 `wire` 或 `reg` 类型
- `inout` 端口必须为 `wire` 类型

```verilog
module port_example (
    input  wire        clk,       // 输入端口，默认 wire
    input  wire [7:0]  data_in,   // 8位输入总线
    output reg  [7:0]  data_out,  // 8位输出总线，reg 类型
    output wire        valid,     // 输出端口，wire 类型
    inout  wire [15:0] bus        // 双向端口，必须为 wire
);

    // 时序逻辑中使用 reg 类型输出
    always @(posedge clk) begin
        data_out <= data_in;
    end

    // 组合逻辑中使用 wire 类型输出
    assign valid = (data_in != 8'h00);

endmodule
```

在 Verilog-2001 及以后的版本中，推荐使用 ANSI 风格的端口声明，将方向和类型合并在一起，使代码更简洁。

## 模块实例化与层次结构

模块实例化是将已定义的模块在上层模块中使用的过程。通过实例化可以构建层次化的设计结构。

```verilog
// 底层模块：4位加法器
module adder_4bit (
    input  wire [3:0] a,
    input  wire [3:0] b,
    input  wire       cin,
    output wire [3:0] sum,
    output wire       cout
);
    assign {cout, sum} = a + b + cin;
endmodule

// 上层模块：8位加法器，由两个4位加法器级联
module adder_8bit (
    input  wire [7:0] a,
    input  wire [7:0] b,
    input  wire       cin,
    output wire [7:0] sum,
    output wire       cout
);
    wire carry;  // 内部连线

    // 低4位加法器实例
    adder_4bit u_low (
        .a    (a[3:0]),
        .b    (b[3:0]),
        .cin  (cin),
        .sum  (sum[3:0]),
        .cout (carry)
    );

    // 高4位加法器实例
    adder_4bit u_high (
        .a    (a[7:4]),
        .b    (b[7:4]),
        .cin  (carry),
        .sum  (sum[7:4]),
        .cout (cout)
    );
endmodule
```

实例化时推荐使用**端口名称关联**（如上例中的 `.a(a[3:0])`），而非位置关联。端口名称关联的优点是：
- 代码可读性更强
- 不受端口顺序影响
- 便于维护和修改

## 参数化模块

参数化模块允许在实例化时通过参数定制模块的行为，提高代码的复用性。使用 `parameter` 关键字定义参数。

```verilog
// 参数化宽度的乘法器
module multiplier #(
    parameter WIDTH = 8
) (
    input  wire [WIDTH-1:0] a,
    input  wire [WIDTH-1:0] b,
    output wire [2*WIDTH-1:0] product
);
    assign product = a * b;
endmodule

// 使用默认参数实例化（8位）
multiplier u_mult8 (
    .a       (data_a),
    .b       (data_b),
    .product (result_8)
);

// 通过参数覆盖实例化（16位）
multiplier #(
    .WIDTH(16)
) u_mult16 (
    .a       (data_a_16),
    .b       (data_b_16),
    .product (result_16)
);
```

`defparam` 也可以用于参数覆盖，但在综合工具中不推荐使用，建议使用 `#()` 语法。

```verilog
// 参数化 FIFO 模块
module sync_fifo #(
    parameter DATA_WIDTH = 32,
    parameter ADDR_WIDTH = 8,
    parameter DEPTH      = (1 << ADDR_WIDTH)
) (
    input  wire                  clk,
    input  wire                  rst_n,
    input  wire                  wr_en,
    input  wire [DATA_WIDTH-1:0] wr_data,
    input  wire                  rd_en,
    output reg  [DATA_WIDTH-1:0] rd_data,
    output wire                  full,
    output wire                  empty
);
    // 内部存储器
    reg [DATA_WIDTH-1:0] mem [0:DEPTH-1];
    reg [ADDR_WIDTH:0] wr_ptr, rd_ptr;

    assign full  = (wr_ptr[ADDR_WIDTH] != rd_ptr[ADDR_WIDTH]) &&
                   (wr_ptr[ADDR_WIDTH-1:0] == rd_ptr[ADDR_WIDTH-1:0]);
    assign empty = (wr_ptr == rd_ptr);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wr_ptr <= 0;
        end else if (wr_en && !full) begin
            mem[wr_ptr[ADDR_WIDTH-1:0]] <= wr_data;
            wr_ptr <= wr_ptr + 1;
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rd_ptr  <= 0;
            rd_data <= 0;
        end else if (rd_en && !empty) begin
            rd_data <= mem[rd_ptr[ADDR_WIDTH-1:0]];
            rd_ptr  <= rd_ptr + 1;
        end
    end
endmodule
```

参数化设计使得同一个模块可以适应不同的数据宽度和深度需求，是 FPGA/ASIC 设计中提高代码复用率的重要手段。
