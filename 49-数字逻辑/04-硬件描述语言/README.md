# 04-硬件描述语言

## 1. HDL 概述

### 1.1 什么是硬件描述语言

硬件描述语言（HDL）用于描述数字电路的结构、行为和功能，支持从算法级到门级的多个抽象层次。

### 1.2 Verilog vs VHDL

| 特性 | Verilog | VHDL |
|------|---------|------|
| 起源 | 1984年，Gateway Design | 1987年，美国国防部 |
| 语法风格 | 类似 C 语言 | 类似 Ada/Pascal |
| 类型系统 | 弱类型，隐式转换 | 强类型，显式转换 |
| 大小写敏感 | 是 | 否 |
| 学习曲线 | 较平缓 | 较陡峭 |
| 主要应用 | ASIC/FPGA（工业界主流） | 航空航天、军事、欧洲 |
| 标准 | IEEE 1364 | IEEE 1076 |

### 1.3 设计层次

```
系统级（算法描述）
    ↓
RTL 级（寄存器传输级）  ← 综合的起点
    ↓
门级（逻辑门实现）
    ↓
开关级（晶体管级）
```

---

## 2. Verilog 基础语法

### 2.1 模块（module）

```verilog
module and_gate(
    input  a,
    input  b,
    output y
);
    assign y = a & b;
endmodule
```

### 2.2 数据类型

**线网类型（Net）：**
- `wire`：最常用，表示信号连线
- `tri`：三态总线
- 表示硬件中的物理连线，不能存储值

**寄存器类型（Reg）：**
- `reg`：在 always 块中被赋值的变量
- `integer`：32位有符号整数
- `parameter`：常量参数

**注意：** `reg` 不一定映射为物理寄存器，它只是 Verilog 的语法概念。

### 2.3 运算符

| 类别 | 运算符 | 说明 |
|------|--------|------|
| 算术 | +, -, *, /, % | 加减乘除取模 |
| 逻辑 | &&, \|\|, ! | 逻辑与或非 |
| 按位 | &, \|, ~, ^, ~^ | 按位与或非异或 |
| 移位 | <<, >> | 逻辑移位 |
| 归约 | &, ~&, \|, ~\|, ^, ~^ | 单操作数各位运算 |
| 关系 | <, >, <=, >= | 比较 |
| 条件 | ? : | 三目运算符 |
| 拼接 | {,} | 位拼接 |

### 2.4 赋值方式

**连续赋值（assign）：**
- 描述组合逻辑
- 右端变化立即影响左端
```verilog
assign y = a & b;
```

**过程赋值（always 块内）：**
- 阻塞赋值 `=`：顺序执行，常用于组合逻辑
- 非阻塞赋值 `<=`：并行执行，常用于时序逻辑

---

## 3. 组合逻辑的 Verilog 描述

### 3.1 使用 assign 语句

```verilog
module mux4to1(
    input  [3:0] d,
    input  [1:0] sel,
    output       y
);
    assign y = d[sel];
endmodule
```

### 3.2 使用 always 块

```verilog
module mux4to1(
    input  [3:0] d,
    input  [1:0] sel,
    output reg   y
);
    always @(*) begin
        case (sel)
            2'b00: y = d[0];
            2'b01: y = d[1];
            2'b10: y = d[2];
            2'b11: y = d[3];
        endcase
    end
endmodule
```

### 3.3 常用组合逻辑描述模板

**多路选择器：** 使用 `case` 或 `assign y = sel ? d1 : d0`

**译码器：** 使用 `case` 或 `assign` + 位选择

**优先编码器：** 使用 `casex` 或 `if-else if` 链

---

## 4. 时序逻辑的 Verilog 描述

### 4.1 D 触发器

```verilog
module d_ff(
    input      clk,
    input      d,
    output reg q
);
    always @(posedge clk) begin
        q <= d;
    end
endmodule
```

### 4.2 带异步复位的触发器

```verilog
module d_ff_reset(
    input      clk,
    input      rst_n,  // 低电平异步复位
    input      d,
    output reg q
);
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            q <= 1'b0;
        else
            q <= d;
    end
endmodule
```

### 4.3 带同步复位的触发器

```verilog
module d_ff_sync(
    input      clk,
    input      rst_n,
    input      d,
    output reg q
);
    always @(posedge clk) begin
        if (!rst_n)
            q <= 1'b0;
        else
            q <= d;
    end
endmodule
```

### 4.4 计数器

```verilog
module counter #(parameter N = 4)(
    input            clk,
    input            rst_n,
    output reg [N-1:0] count
);
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            count <= 0;
        else
            count <= count + 1;
    end
endmodule
```

### 4.5 移位寄存器

```verilog
module shift_reg #(parameter N = 8)(
    input            clk,
    input            rst_n,
    input            sin,
    output reg [N-1:0] q
);
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            q <= 0;
        else
            q <= {q[N-2:0], sin};
    end
endmodule
```

---

## 5. Testbench 编写

### 5.1 Testbench 结构

```verilog
module tb_and_gate;
    reg  a, b;
    wire y;

    // 实例化被测模块
    and_gate uut (.a(a), .b(b), .y(y));

    initial begin
        // 初始化
        a = 0; b = 0;
        #10 a = 0; b = 1;
        #10 a = 1; b = 0;
        #10 a = 1; b = 1;
        #10 $finish;
    end

    // 波形输出
    initial begin
        $dumpfile("wave.vcd");
        $dumpvars(0, tb_and_gate);
    end
endmodule
```

### 5.2 时钟生成

```verilog
// 方法一：initial 块
reg clk;
initial clk = 0;
always #5 clk = ~clk;  // 周期 10ns

// 方法二：简洁写法
reg clk = 0;
always #5 clk = ~clk;
```

### 5.3 常用系统任务

| 任务 | 功能 |
|------|------|
| `$display` | 立即打印（类似 printf） |
| `$monitor` | 监视信号变化时打印 |
| `$time` | 获取当前仿真时间 |
| `$random` | 生成随机数 |
| `$finish` | 结束仿真 |
| `$dumpfile` | 指定波形文件名 |
| `$dumpvars` | 转储变量到波形文件 |
| `$readmemh` | 从文件读取初始化数据 |

---

## 6. 有限状态机的 Verilog 实现

### 6.1 三段式 FSM 模板

```verilog
module fsm_101_detector(
    input  clk,
    input  rst_n,
    input  din,
    output reg dout
);
    // 状态编码
    parameter S0 = 2'b00, S1 = 2'b01, S2 = 2'b10;
    reg [1:0] state, next_state;

    // 第一段：状态寄存器
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            state <= S0;
        else
            state <= next_state;
    end

    // 第二段：次态逻辑（组合逻辑）
    always @(*) begin
        case (state)
            S0: next_state = din ? S1 : S0;
            S1: next_state = din ? S1 : S2;
            S2: next_state = din ? S1 : S0;
            default: next_state = S0;
        endcase
    end

    // 第三段：输出逻辑
    always @(*) begin
        dout = (state == S2) && din;
    end
endmodule
```

### 6.2 独热编码（One-Hot）FSM

```verilog
parameter S0 = 4'b0001, S1 = 4'b0010,
          S2 = 4'b0100, S3 = 4'b1000;
```

优点：次态逻辑简单，速度快；缺点：触发器使用较多。

### 6.3 FSM 编写注意事项

- 次态逻辑和输出逻辑使用阻塞赋值（`=`），组合逻辑 `always @(*)`
- 状态寄存器使用非阻塞赋值（`<=`），时序逻辑 `always @(posedge clk)`
- 始终提供 `default` 分支避免锁存器
- 三段式写法便于综合工具识别

---

## 7. FPGA 架构与开发流程

### 7.1 FPGA 基本架构

```
┌──────────────────────────────────────┐
│  可编程 I/O 单元                      │
│  ┌──────┐  ┌──────┐  ┌──────┐       │
│  │ CLB  │──│ CLB  │──│ CLB  │       │
│  └──┬───┘  └──┬───┘  └──┬───┘       │
│     │可编程互连│         │           │
│  ┌──┴───┐  ┌──┴───┐  ┌──┴───┐       │
│  │ CLB  │──│ CLB  │──│ CLB  │       │
│  └──────┘  └──────┘  └──────┘       │
│  [BRAM] [DSP Slice] [PLL/MMCM]     │
└──────────────────────────────────────┘
```

**CLB（可配置逻辑块）：** 包含 LUT（查找表）、触发器和多路选择器。

**LUT（查找表）：** n 输入 LUT 可实现任意 n 变量布尔函数（本质是 2^n × 1 的 SRAM）。

**BRAM：** 嵌入式块存储器，用于实现 FIFO、RAM、ROM。

**DSP Slice：** 专用硬件乘法器/累加器。

### 7.2 开发流程

```
HDL 代码编写
    ↓
功能仿真（Simulation）
    ↓
综合（Synthesis）：HDL → 门级网表
    ↓
实现（Implementation）
   ├── 翻译（Translate）
   ├── 映射（Map）：门级网表 → FPGA 资源
   └── 布局布线（Place & Route）
    ↓
时序分析（Timing Analysis）
    ↓
生成比特流（Bitstream）
    ↓
下载到 FPGA
```

### 7.3 与 ASIC 设计的区别

| 特性 | FPGA | ASIC |
|------|------|------|
| 开发周期 | 短（数周~数月） | 长（数月~数年） |
| 单片成本 | 较高（量产时） | 低（量产时） |
| 灵活性 | 可重编程 | 流片后不可更改 |
| 性能上限 | 较低 | 更高 |
| 适用场景 | 原型验证、小批量 | 大批量、高性能 |

---

## 8. 常用 EDA 工具简介

### 8.1 综合与仿真工具

| 工具 | 厂商 | 用途 |
|------|------|------|
| Vivado | Xilinx/AMD | FPGA 综合、实现、仿真 |
| Quartus Prime | Intel/Altera | FPGA 综合、实现 |
| ModelSim | Siemens | 仿真 |
| VCS | Synopsys | 仿真 |
| Design Compiler | Synopsys | ASIC 综合 |
| ISE | Xilinx（旧） | 旧版 FPGA 工具 |

### 8.2 开源工具

| 工具 | 用途 |
|------|------|
| Icarus Verilog | Verilog 编译与仿真 |
| GTKWave | 波形查看器 |
| Yosys | 开源综合工具 |
| Verilator | 高速 Verilog 仿真器 |
| Cocotb | Python 写 testbench |
| Edalize | EDA 工具流程管理 |

### 8.3 仿真工具的基本使用

```bash
# Icarus Verilog 示例
iverilog -o sim tb_design.v design.v   # 编译
vvp sim                                  # 运行仿真
gtkwave wave.vcd                         # 查看波形
```

---

## 参考资料

- Morris Mano, *Digital Design*, 5th Edition
- Samir Palnitkar, *Verilog HDL: A Guide to Digital Design and Synthesis*
- Stuart Sutherland, *Verilog-2001: A Guide to the New Features*
- IEEE Standard 1364-2005 (Verilog)
- Xilinx UG901: Vivado Design Suite User Guide - Synthesis
