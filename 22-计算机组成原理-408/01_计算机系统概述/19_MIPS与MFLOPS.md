# 20_MIPS与MFLOPS

## 核心概念

### MIPS

**MIPS（Million Instructions Per Second）**：每秒执行的百万条指令数。

$$
\text{MIPS} = \frac{IC}{T_{\text{CPU}} \times 10^6} = \frac{f_{\text{主频}}}{CPI \times 10^6}
$$

其中：
- $IC$：指令总数
- $T_{\text{CPU}}$：CPU执行时间（s）
- $f$：主频（Hz）
- $CPI$：每条指令的平均时钟周期数

**特点**：
- MIPS越大，性能越好
- 适用于评价整数运算密集的程序
- 不同指令集的计算机之间MIPS不可直接比较

### MFLOPS

**MFLOPS（Million Floating-point Operations Per Second）**：每秒百万次浮点运算次数。

$$
\text{MFLOPS} = \frac{\text{浮点运算次数}}{T_{\text{CPU}} \times 10^6}
$$

**特点**：
- MFLOPS越大，浮点运算性能越好
- 主要用于科学计算、图形处理等浮点密集型应用
- 不能用于比较不同类型应用的性能

### MIPS与MFLOPS的对比

| 对比维度 | MIPS | MFLOPS |
|---------|------|--------|
| 全称 | Million Instructions Per Second | Million Floating-point Operations Per Second |
| 衡量对象 | 指令执行速度 | 浮点运算速度 |
| 适用场景 | 整数运算、通用计算 | 科学计算、图形处理 |
| 可比性 | 同一指令集内可比 | 同类应用内可比 |
| 局限性 | 不能跨指令集比较 | 不能评价整数运算性能 |

## 原理分析

### MIPS的推导

$$
\text{MIPS} = \frac{IC}{T_{\text{CPU}} \times 10^6} = \frac{IC}{\frac{IC \times CPI}{f} \times 10^6} = \frac{f}{CPI \times 10^6}
$$

**例题**：某CPU主频为2GHz，平均CPI为1.5，求MIPS。

$$
\text{MIPS} = \frac{2 \times 10^9}{1.5 \times 10^6} = \frac{2000}{1.5} \approx 1333.33
$$

### MIPS的局限性

**问题1**：不同指令集的指令复杂度不同
- CISC（复杂指令集）：一条指令可能完成多步操作
- RISC（简单指令集）：一条指令完成一步操作
- 两者的MIPS不能直接比较

**例**：完成一个乘法运算：
- CISC可能用1条指令，CPI=10
- RISC可能用10条指令，每条CPI=1
- 总周期数相同，但MIPS值不同

**问题2**：MIPS高不代表实际性能好
- 有些CPU指令简单但数量多
- MIPS = $f / (CPI \times 10^6)$，高主频低CPI就会得到高MIPS
- 但如果执行的指令数量更多，实际执行时间可能更长

### MFLOPS的计算

**例题**：某程序需要执行 $5 \times 10^9$ 次浮点运算，在某CPU上运行耗时2s，求MFLOPS。

$$
\text{MFLOPS} = \frac{5 \times 10^9}{2 \times 10^6} = 2500
$$

## 直观理解

### 生活类比

**MIPS**就像工厂每小时生产的"标准件"数量：
- 不同工厂的"标准件"大小不同
- A厂每小时做100个大件，B厂做200个小件
- 不能说B厂更快（可能A厂的总工作量更大）

**MFLOPS**就像每小时完成的"计算题"数量：
- 只统计数学计算题（浮点运算）
- 不统计语文作业（整数运算）
- 适用于评价"理科班"（科学计算）的效率

### 记忆技巧

MIPS = 主频 / (CPI × $10^6$) → 「频除CPI乘百万」
MFLOPS = 浮点运算次数 / 时间 / $10^6$ → 「浮运除时除百万」

## 知识关联

### 常见性能指标汇总

| 指标 | 公式 | 单位 | 越大越好? |
|------|------|------|----------|
| CPU时间 | $IC \times CPI / f$ | s | 否 |
| MIPS | $f / (CPI \times 10^6)$ | 百万条/秒 | 是 |
| MFLOPS | 浮点运算数 / ($T \times 10^6$) | 百万次/秒 | 是 |
| CPI | 总周期数 / 指令数 | 周期/指令 | 否 |

### 易错陷阱

1. **MIPS不能跨指令集比较**：RISC和CISC的MIPS不可比
2. **MFLOPS只适用于浮点运算**：整数运算密集的程序用MIPS评价
3. **MIPS和主频不是简单的正比关系**：CPI也会影响MIPS
4. **MFLOPS和MIPS不能直接换算**：因为不同程序的浮点运算比例不同
5. **不要把MIPS当作唯一性能指标**：它只是一个参考值
