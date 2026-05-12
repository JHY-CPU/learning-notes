# 100_CPU设计实例

## 核心概念

- 从**需求分析**到**数据通路**到**控制信号**的完整 CPU 设计流程
- 以一个简化 CPU 为例，覆盖取指、译码、执行、访存、写回
- 408 大题可能涉及的部分设计

## 原理分析

### 一、设计目标

设计一个支持以下指令的简单 CPU：
- **算术逻辑**：ADD, SUB, AND, OR
- **访存**：LD（Load）, ST（Store）
- **转移**：BEQ（Branch if Equal）

### 二、数据通路设计

```
┌─────────────────────────────────────────────┐
│                    CPU                        │
│                                              │
│  ┌─────┐    ┌──────┐    ┌─────┐             │
│  │ PC  │───→│  +4  │───→│ MUX │──→ PC_next  │
│  └──┬──┘    └──────┘    └──┬──┘             │
│     │                      │                 │
│     ↓                      │ branch_target   │
│  ┌──────┐                  │                 │
│  │ IMEM │──→ IR            │                 │
│  └──────┘                  │                 │
│     ↓                      │                 │
│  ┌──────┐                  │                 │
│  │ Reg  │←── Read Rs1, Rs2 │                 │
│  │ File │──→ A, B          │                 │
│  └──────┘                  │                 │
│     ↓        ┌──────┐      │                 │
│  ┌──────┐   │  +   │←──────┤                 │
│  │ MUX  │──→│ ALU  │──→ ALU_out             │
│  └──────┘   └──────┘      │                 │
│  A/Imm       ↑             │                 │
│              │             │                 │
│  ┌──────┐   │             │                 │
│  │ DMem │←──ALU_out       │                 │
│  │      │──→ Mem_data     │                 │
│  └──────┘                  │                 │
│     ↓                      │                 │
│  ┌──────┐                  │                 │
│  │ MUX  │←── ALU_out/Mem_data               │
│  └──┬───┘                  │                 │
│     ↓                      │                 │
│  Reg[ Rd ]←── 写回数据     │                 │
└─────────────────────────────────────────────┘
```

### 三、各指令的微操作与控制信号

#### ADD R1, R2, R3

| 周期 | 微操作 | 控制信号 |
|------|--------|---------|
| IF | PC→MAR, Read, MDR→IR, PC+4→PC | PCout, Read, IRin, PC+1 |
| ID | IR[rs1]→Reg, IR[rs2]→Reg | RegRead |
| EX | A + B → ALU_out | ALUop=ADD |
| MEM | （无操作） | - |
| WB | ALU_out → Reg[rd] | RegWrite |

#### LD R1, offset(R2)

| 周期 | 微操作 | 控制信号 |
|------|--------|---------|
| EX | A + Imm(offset) → ALU_out | ALUop=ADD, ImmSel |
| MEM | ALU_out→MAR, Read, MDR←Mem | MARin, Read, MDRin |
| WB | MDR → Reg[rd] | MemtoReg, RegWrite |

#### ST R1, offset(R2)

| 周期 | 微操作 | 控制信号 |
|------|--------|---------|
| EX | A + Imm → ALU_out | ALUop=ADD, ImmSel |
| MEM | ALU_out→MAR, B→MDR, Write | MARin, MDRin, Write |
| WB | （无操作） | - |

#### BEQ R1, R2, offset

| 周期 | 微操作 | 控制信号 |
|------|--------|---------|
| EX | A - B, 比较是否为 0 | ALUop=SUB, Zero flag |
| MEM | PC+4+Imm → PC（若 Zero=1）| PCSrc, Branch |
| WB | （无操作） | - |

### 四、控制信号汇总

| 信号 | 功能 |
|------|------|
| PCout | PC 送总线 |
| IRin | 总线数据送 IR |
| RegRead | 读寄存器 |
| RegWrite | 写寄存器 |
| ALUop | ALU 操作类型 |
| MemRead | 读存储器 |
| MemWrite | 写存储器 |
| MemtoReg | 存储器数据送寄存器 |
| PCSrc | PC 来源选择（顺序/转移） |
| ImmSel | 立即数选择 |

## 代码/模拟

### Verilog实现简化CPU核心模块

```verilog
// ============================================
// 简化CPU核心模块 - 适用于408考研理解数据通路
// 支持: ADD, SUB, AND, OR, LD, ST, BEQ
// ============================================

// --- ALU模块 ---
module alu(
    input  [31:0] a, b,
    input  [2:0]  alu_ctrl,    // ALU控制信号
    output reg [31:0] result,
    output zero                // 零标志（用于BEQ判断）
);
    always @(*) begin
        case (alu_ctrl)
            3'b000: result = a + b;        // ADD
            3'b001: result = a - b;        // SUB
            3'b010: result = a & b;        // AND
            3'b011: result = a | b;        // OR
            3'b100: result = a + b;        // LD/ST计算地址
            default: result = 32'b0;
        endcase
    end
    assign zero = (result == 32'b0);
endmodule

// --- 寄存器堆 ---
module register_file(
    input         clk,
    input  [4:0]  rs1, rs2, rd,   // 源寄存器1/2, 目标寄存器
    input  [31:0] wd,              // 写入数据
    input         we,              // 写使能
    output [31:0] rd1, rd2         // 读出数据
);
    reg [31:0] regs [0:31];       // 32个32位寄存器

    assign rd1 = (rs1 != 0) ? regs[rs1] : 32'b0;  // x0恒为0
    assign rd2 = (rs2 != 0) ? regs[rs2] : 32'b0;

    always @(posedge clk) begin
        if (we && rd != 0)
            regs[rd] <= wd;
    end
endmodule

// --- 指令存储器 ---
module instr_mem(
    input  [31:0] addr,
    output [31:0] instr
);
    reg [31:0] mem [0:255];       // 256条指令
    assign mem[addr[31:2]];       // 字节寻址转字寻址
    assign instr = mem[addr[9:2]];
endmodule

// --- 数据存储器 ---
module data_mem(
    input         clk,
    input  [31:0] addr, wd,
    input         mem_write, mem_read,
    output [31:0] rd
);
    reg [31:0] mem [0:255];
    always @(posedge clk)
        if (mem_write) mem[addr[9:2]] <= wd;
    assign rd = mem_read ? mem[addr[9:2]] : 32'b0;
endmodule

// --- 控制单元 ---
module control_unit(
    input  [6:0] opcode,
    output reg reg_write, mem_read, mem_write,
    output reg branch, alu_src, mem_to_reg,
    output reg [1:0] alu_op
);
    always @(*) begin
        // 默认值
        {reg_write, mem_read, mem_write, branch,
         alu_src, mem_to_reg, alu_op} = 0;
        case (opcode)
            7'b0110011: begin  // R-type (ADD/SUB/AND/OR)
                reg_write = 1; alu_op = 2'b10;
            end
            7'b0000011: begin  // LD
                reg_write = 1; mem_read = 1;
                alu_src = 1; mem_to_reg = 1; alu_op = 2'b00;
            end
            7'b0100011: begin  // ST
                mem_write = 1; alu_src = 1; alu_op = 2'b00;
            end
            7'b1100011: begin  // BEQ
                branch = 1; alu_op = 2'b01;
            end
        endcase
    end
endmodule
```

### Python模拟控制信号生成

```python
"""控制信号生成模拟 - 对应408考试中的控制信号分析题"""

def generate_control_signals(opcode):
    """
    根据操作码生成控制信号（对应Verilog控制单元的行为）
    返回: (RegWrite, ALUSrc, MemRead, MemWrite, MemToReg, Branch, ALUOp)
    """
    signals = {
        # R-type: ADD, SUB, AND, OR
        'R': (1, 0, 0, 0, 0, 0, 0b10),
        # I-type: LD
        'LD': (1, 1, 1, 0, 1, 0, 0b00),
        # S-type: ST
        'ST': (0, 1, 0, 1, 0, 0, 0b00),
        # B-type: BEQ
        'BEQ': (0, 0, 0, 0, 0, 1, 0b01),
    }

    names = ['RegWrite', 'ALUSrc', 'MemRead', 'MemWrite',
             'MemToReg', 'Branch', 'ALUOp']

    if opcode not in signals:
        print(f"未知操作码: {opcode}")
        return None

    vals = signals[opcode]
    print(f"\n指令类型: {opcode}")
    print(f"{'信号':<12} {'值':>3}")
    print("-" * 18)
    for name, val in zip(names, vals):
        print(f"{name:<12} {val:>3}")
    return vals

print("=== 控制信号生成 ===")
generate_control_signals('R')    # ADD R1, R2, R3
generate_control_signals('LD')   # LD R1, offset(R2)
generate_control_signals('ST')   # ST R1, offset(R2)
generate_control_signals('BEQ')  # BEQ R1, R2, offset
```

## 知识关联

- 数据通路设计：第 22-26 节
- 微操作分析：第 30-34 节
- 控制器设计：第 28-55 节
- 指令格式：第 8 章
- 408 大题：常考数据通路 + 控制信号分析
