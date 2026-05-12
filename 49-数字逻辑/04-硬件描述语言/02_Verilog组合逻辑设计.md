# Verilog组合逻辑设计

## 连续赋值语句 assign

组合逻辑的输出仅取决于当前的输入值，不依赖于任何历史状态。在 Verilog 中，组合逻辑主要通过两种方式描述：`assign` 连续赋值语句和 `always @(*)` 过程块。

`assign` 语句用于对 `wire` 类型信号进行连续赋值，当右侧表达式的值发生变化时，左侧信号立即更新。这是描述简单组合逻辑最直接的方式。

```verilog
module comb_logic_assign (
    input  wire [3:0] a,
    input  wire [3:0] b,
    input  wire       sel,
    output wire [3:0] y,
    output wire       eq,
    output wire       gt
);
    // 基本逻辑运算
    assign y  = sel ? a : b;          // 2选1多路选择器
    assign eq = (a == b);             // 相等比较
    assign gt = (a > b);              // 大于比较

endmodule
```

`assign` 支持的运算符包括：
- **算术运算符**：`+`、`-`、`*`、`/`、`%`
- **逻辑运算符**：`&&`、`||`、`!`
- **位运算符**：`&`、`|`、`^`、`~`
- **关系运算符**：`<`、`>`、`<=`、`>=`、`==`、`!=`
- **移位运算符**：`<<`、`>>`
- **拼接运算符**：`{}`、`{{}}`
- **条件运算符**：`? :`

## always 块描述组合逻辑

使用 `always @(*)` 块可以描述更复杂的组合逻辑。敏感列表中的 `*` 表示自动推断所有输入信号，任何输入变化都会触发块内语句的执行。在 `always` 块中，输出信号必须声明为 `reg` 类型。

```verilog
module mux_4to1 (
    input  wire [1:0] sel,
    input  wire [7:0] din0,
    input  wire [7:0] din1,
    input  wire [7:0] din2,
    input  wire [7:0] din3,
    output reg  [7:0] dout
);
    // 使用 case 语句描述 4 选 1 多路选择器
    always @(*) begin
        case (sel)
            2'b00:   dout = din0;
            2'b01:   dout = din1;
            2'b10:   dout = din2;
            2'b11:   dout = din3;
            default: dout = 8'h00;
        endcase
    end
endmodule
```

在组合逻辑的 `always` 块中需要注意：
- 所有输入信号必须出现在敏感列表中（使用 `*` 可自动处理）
- 所有分支必须完整，避免产生锁存器（latch）
- 使用 `if-else` 时必须有 `else` 分支

```verilog
// 使用 if-else 描述优先编码器
module priority_encoder (
    input  wire [7:0] din,
    output reg  [2:0] code,
    output reg        valid
);
    always @(*) begin
        valid = 1'b1;
        if (din[7])       code = 3'd7;
        else if (din[6])  code = 3'd6;
        else if (din[5])  code = 3'd5;
        else if (din[4])  code = 3'd4;
        else if (din[3])  code = 3'd3;
        else if (din[2])  code = 3'd2;
        else if (din[1])  code = 3'd1;
        else if (din[0])  code = 3'd0;
        else begin
            code  = 3'd0;
            valid = 1'b0;
        end
    end
endmodule
```

## 常用组合逻辑电路设计

### 译码器

译码器（Decoder）将 n 位输入编码转换为 2^n 个输出中的一个有效输出。

```verilog
module decoder_3to8 (
    input  wire [2:0] din,
    input  wire       en,
    output reg  [7:0] dout
);
    always @(*) begin
        if (en) begin
            case (din)
                3'd0: dout = 8'b0000_0001;
                3'd1: dout = 8'b0000_0010;
                3'd2: dout = 8'b0000_0100;
                3'd3: dout = 8'b0000_1000;
                3'd4: dout = 8'b0001_0000;
                3'd5: dout = 8'b0010_0000;
                3'd6: dout = 8'b0100_0000;
                3'd7: dout = 8'b1000_0000;
            endcase
        end else begin
            dout = 8'h00;
        end
    end
endmodule
```

### 编码器

编码器（Encoder）是译码器的逆操作，将 2^n 个输入编码为 n 位输出。

```verilog
module encoder_8to3 (
    input  wire [7:0] din,
    output reg  [2:0] code,
    output reg        valid
);
    always @(*) begin
        valid = 1'b1;
        casez (din)          // casez 允许 ? 或 z 作为通配符
            8'b1???_????: code = 3'd7;
            8'b01??_????: code = 3'd6;
            8'b001?_????: code = 3'd5;
            8'b0001_????: code = 3'd4;
            8'b0000_1???: code = 3'd3;
            8'b0000_01??: code = 3'd2;
            8'b0000_001?: code = 3'd1;
            8'b0000_0001: code = 3'd0;
            default: begin
                code  = 3'd0;
                valid = 1'b0;
            end
        endcase
    end
endmodule
```

### ALU 设计

算术逻辑单元（ALU）是处理器的核心部件，根据操作码执行不同的算术和逻辑运算。

```verilog
module alu_8bit (
    input  wire [7:0] a,
    input  wire [7:0] b,
    input  wire [2:0] op,
    output reg  [7:0] result,
    output wire       zero,
    output wire       carry
);
    reg [8:0] temp;  // 9位用于捕获进位

    always @(*) begin
        case (op)
            3'b000: temp = {1'b0, a} + {1'b0, b};     // 加法
            3'b001: temp = {1'b0, a} - {1'b0, b};     // 减法
            3'b010: temp = {1'b0, a & b};              // 按位与
            3'b011: temp = {1'b0, a | b};              // 按位或
            3'b100: temp = {1'b0, a ^ b};              // 按位异或
            3'b101: temp = {1'b0, ~a};                 // 按位取反
            3'b110: temp = {1'b0, a << b[2:0]};        // 左移
            3'b111: temp = {1'b0, a >> b[2:0]};        // 右移
        endcase
        result = temp[7:0];
    end

    assign carry = temp[8];
    assign zero  = (result == 8'h00);
endmodule
```

在组合逻辑设计中，最常见的错误是遗漏某些分支导致综合工具推断出锁存器。避免锁存器的方法：
1. 在 `if` 语句中始终提供 `else` 分支
2. 在 `case` 语句中始终包含 `default` 分支
3. 在 `always` 块开头为所有输出信号赋默认值
