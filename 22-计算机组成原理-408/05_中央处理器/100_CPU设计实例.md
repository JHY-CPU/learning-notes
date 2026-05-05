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
└─────────────────────────────────────────────┘```

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

## 知识关联

- 数据通路设计：第 22-26 节
- 微操作分析：第 30-34 节
- 控制器设计：第 28-55 节
- 指令格式：第 8 章
- 408 大题：常考数据通路 + 控制信号分析
