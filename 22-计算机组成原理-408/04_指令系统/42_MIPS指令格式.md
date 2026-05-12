# 43_MIPS指令格式

## 核心概念

### MIPS基本特征
- **32位定长指令**
- **Load/Store结构**
- **32个通用寄存器**（$0\sim$31），$0恒为0
- **三种指令格式**：R型、I型、J型

### R型（Register）指令

用于**寄存器-寄存器运算**。

```
┌────────┬─────┬─────┬─────┬──────┬────────┐
│opcode  │ rs  │ rt  │ rd  │shamt │ funct  │
│  6位   │ 5位 │ 5位 │ 5位 │ 5位  │  6位   │
└────────┴─────┴─────┴─────┴──────┴────────┘
 31    26 25  21 20  16 15  11 10   6 5     0
```

| 字段 | 含义 |
|:---|:---|
| **opcode** | 操作码，R型固定为000000 |
| **rs** | 源寄存器1 |
| **rt** | 源寄存器2 |
| **rd** | 目的寄存器 |
| **shamt** | 移位量（shift amount） |
| **funct** | 功能码，指定具体操作 |

**示例**：`ADD $t0, $t1, $t2` → $t0 \leftarrow $t1 + $t2$
```
opcode=000000, rs=$t1, rt=$t2, rd=$t0, shamt=00000, funct=100000
```

### I型（Immediate）指令

用于**立即数操作、LOAD/STORE、分支**。

```
┌────────┬─────┬─────┬──────────────────┐
│opcode  │ rs  │ rt  │   immediate      │
│  6位   │ 5位 │ 5位 │     16位         │
└────────┴─────┴─────┴──────────────────┘
 31    26 25  21 20  16 15               0
```

| 字段 | 含义 |
|:---|:---|
| **opcode** | 操作码（非0） |
| **rs** | 源寄存器 |
| **rt** | 目的寄存器（LOAD）或源寄存器（STORE） |
| **immediate** | 16位立即数/偏移量 |

**示例**：
- `ADDI $t0, $t1, 100` → $t0 \leftarrow $t1 + 100
- `LW $t0, 100($t1)` → $t0 \leftarrow M[$t1 + 100]
- `BEQ $t0, $t1, offset` → if $t0==$t1 then PC ← PC + offset

### J型（Jump）指令

用于**无条件跳转**。

```
┌────────┬──────────────────────────────┐
│opcode  │        address (26位)        │
│  6位   │                              │
└────────┴──────────────────────────────┘
 31    26 25                            0
```

跳转目标：$PC = (PC_{31:28}) \| (address << 2)$

**示例**：`J target`
```
opcode=000010, address=目标地址的高26位（右移2位）
```

## 原理分析

### 为什么R型的opcode = 0

R型指令的opcode固定为000000，用**funct字段**区分具体操作：

$$\text{有效操作码} = opcode + funct = 0 + funct$$

这使得：
- 6位opcode + 6位funct = 12位可区分操作
- 实际可表示远多于 $2^6 = 64$ 种操作

### I型立即数的符号扩展

16位立即数在使用时需要**符号扩展**到32位：

$$有符号扩展：高16位 = 符号位的复制$$
$$无符号扩展：高16位 = 0$$

| 指令 | 扩展方式 |
|:---|:---|
| ADDI、LW、SW | 符号扩展 |
| ANDI、ORI | 零扩展 |

### J型跳转范围

26位地址 + 左移2位（指令按4字节对齐）= 28位跳转范围

$$跳转目标 = (PC+4)_{31:28} \| address_{25:0} \| 00$$

范围：以PC高4位为基的256MB区域内。

### MIPS寄存器约定

| 编号 | 名称 | 用途 |
|:---:|:---|:---|
| $0 | $zero | 恒为0 |
| $1 | $at | 汇编器临时 |
| $2-$3 | $v0-$v1 | 函数返回值 |
| $4-$7 | $a0-$a3 | 函数参数 |
| $8-$15 | $t0-$t7 | 临时寄存器 |
| $16-$23 | $s0-$s7 | 保存寄存器 |
| $24-$25 | $t8-$t9 | 临时寄存器 |
| $28 | $gp | 全局指针 |
| $29 | $sp | 堆栈指针 |
| $30 | $fp | 帧指针 |
| $31 | $ra | 返回地址 |

## 直观理解

### 记忆技巧

- **"R型做运算（Reg），I型带立即数（Imm），J型跳远（Jump）"**
- **"R型funct选操作，I型16位立即数，J型26位地址"**
- **"所有指令都是32位"**

## 知识关联

### 跨章节联系
- **第29-30节（CISC/RISC）**：MIPS是RISC的典型代表
- **第6-7节（操作码）**：MIPS用定长操作码（6位）+ funct扩展
- **第13-21节（寻址方式）**：MIPS有立即、直接、寄存器、基址+偏移、相对、寄存器间接等

## 代码/模拟

### MIPS汇编代码示例

```mips
# ============================================
# MIPS汇编示例 - 三种指令格式的实际使用
# 适用于408考研理解指令编码
# ============================================

# --- R型指令示例 ---
# 所有R型指令的 opcode=000000，靠funct区分操作

add  $t0, $t1, $t2    # $t0 = $t1 + $t2  (funct=100000)
sub  $t0, $t1, $t2    # $t0 = $t1 - $t2  (funct=100010)
and  $t0, $t1, $t2    # $t0 = $t1 & $t2  (funct=100100)
or   $t0, $t1, $t2    # $t0 = $t1 | $t2  (funct=100101)
sll  $t0, $t1, 4      # $t0 = $t1 << 4   (funct=000000, shamt=4)
jr   $ra              # PC = $ra          (funct=001000)

# --- I型指令示例 ---
# 带有16位立即数或偏移量

addi $t0, $t1, 100    # $t0 = $t1 + 100          (opcode=001000)
lw   $t0, 8($sp)      # $t0 = Memory[$sp + 8]    (opcode=100011)
sw   $t0, 8($sp)      # Memory[$sp + 8] = $t0    (opcode=101011)
beq  $t0, $t1, label  # if $t0==$t1: PC=PC+4+offset (opcode=000100)
bne  $t0, $t1, label  # if $t0!=$t1: PC=PC+4+offset (opcode=000101)

# --- J型指令示例 ---
# 26位跳转地址

j    target            # PC = (PC高4位)|(target<<2) (opcode=000010)
jal  func              # $ra = PC+4, PC=...        (opcode=000011)
```

### Python模拟MIPS指令编码/解码

```python
"""MIPS指令编码与解码模拟 - 适用于408考研复习"""

# 寄存器名称到编号的映射
REG_MAP = {
    '$zero': 0, '$at': 1, '$v0': 2, '$v1': 3,
    '$a0': 4, '$a1': 5, '$a2': 6, '$a3': 7,
    '$t0': 8, '$t1': 9, '$t2': 10, '$t3': 11,
    '$t4': 12, '$t5': 13, '$t6': 14, '$t7': 15,
    '$s0': 16, '$s1': 17, '$s2': 18, '$s3': 19,
    '$s4': 20, '$s5': 21, '$s6': 22, '$s7': 23,
    '$t8': 24, '$t9': 25, '$gp': 28, '$sp': 29, '$fp': 30, '$ra': 31,
}

# R型指令功能码
R_FUNCT = {
    'add': 0b100000, 'sub': 0b100010,
    'and': 0b100100, 'or':  0b100101,
    'sll': 0b000000,
}

# I型指令操作码
I_OPCODE = {
    'addi': 0b001000, 'lw': 0b100011, 'sw': 0b101011,
    'beq': 0b000100, 'bne': 0b000101,
}

def encode_r_type(rd, rs, rt, shamt, funct):
    """编码R型指令: 32位"""
    opcode = 0  # R型固定为0
    instr = (opcode << 26) | (rs << 21) | (rt << 16) | \
            (rd << 11) | (shamt << 6) | funct
    return instr

def encode_i_type(opcode, rs, rt, immediate):
    """编码I型指令: 32位"""
    # 立即数取低16位（处理负数）
    imm16 = immediate & 0xFFFF
    instr = (opcode << 26) | (rs << 21) | (rt << 16) | imm16
    return instr

def decode_instruction(instr):
    """解码32位MIPS指令"""
    opcode = (instr >> 26) & 0x3F
    rs = (instr >> 21) & 0x1F
    rt = (instr >> 16) & 0x1F
    rd = (instr >> 11) & 0x1F
    shamt = (instr >> 6) & 0x1F
    funct = instr & 0x3F
    imm16 = instr & 0xFFFF
    addr26 = instr & 0x3FFFFFF

    # 符号扩展16位立即数
    if imm16 & 0x8000:
        imm_sext = imm16 - 0x10000
    else:
        imm_sext = imm16

    if opcode == 0:
        # R型指令
        return f"R型: opcode={opcode:06b}, rs={rs}, rt={rt}, rd={rd}, " \
               f"shamt={shamt}, funct={funct:06b}"
    elif opcode in (0b000010, 0b000011):
        # J型指令
        return f"J型: opcode={opcode:06b}, address={addr26:026b}"
    else:
        # I型指令
        return f"I型: opcode={opcode:06b}, rs={rs}, rt={rt}, imm={imm_sext}"

# 编码示例
print("=== MIPS指令编码示例 ===\n")

# ADD $t0, $t1, $t2
instr = encode_r_type(rd=8, rs=9, rt=10, shamt=0, funct=0b100000)
print(f"ADD $t0,$t1,$t2 = {instr:032b} = 0x{instr:08X}")
print(f"  解码: {decode_instruction(instr)}\n")

# ADDI $t0, $t1, 100
instr = encode_i_type(opcode=0b001000, rs=9, rt=8, immediate=100)
print(f"ADDI $t0,$t1,100 = {instr:032b} = 0x{instr:08X}")
print(f"  解码: {decode_instruction(instr)}\n")

# LW $t0, 8($sp)
instr = encode_i_type(opcode=0b100011, rs=29, rt=8, immediate=8)
print(f"LW $t0,8($sp) = {instr:032b} = 0x{instr:08X}")
print(f"  解码: {decode_instruction(instr)}\n")

# BEQ $t0, $t1, offset=-4
instr = encode_i_type(opcode=0b000100, rs=8, rt=9, immediate=-4)
print(f"BEQ $t0,$t1,-4 = {instr:032b} = 0x{instr:08X}")
print(f"  解码: {decode_instruction(instr)}")
```

### 易错陷阱
1. **R型opcode=0不是没有操作码**：操作码在funct字段
2. **$0恒为0**：写入$0的数据会被丢弃
3. **I型偏移按字节计算**：虽然指令4字节对齐
4. **J型跳转范围有限**：256MB区域
