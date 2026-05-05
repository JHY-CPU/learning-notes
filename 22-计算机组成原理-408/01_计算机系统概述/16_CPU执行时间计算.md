# 17_CPU执行时间计算

## 核心概念

### CPU执行时间公式

CPU执行时间（CPU Time）是衡量计算机性能的最基本指标，表示CPU执行一段程序所花费的时间。

$$
T_{\text{CPU}} = \frac{\text{指令数} \times \text{CPI}}{f_{\text{主频}}}
$$

等价形式：

$$
T_{\text{CPU}} = \text{指令数} \times \text{CPI} \times T_{\text{时钟}}
$$

其中：
- 指令数（Instruction Count）：程序中执行的指令总数
- CPI（Cycles Per Instruction）：每条指令的平均时钟周期数
- $f_{\text{主频}}$：CPU的主频（Hz）
- $T_{\text{时钟}} = \frac{1}{f}$：时钟周期（s）

### 三个影响因素

| 因素 | 影响 | 决定者 |
|------|------|--------|
| 指令数 | 程序需要执行多少条指令 | 算法、编程语言、编译器 |
| CPI | 每条指令需要多少时钟周期 | CPU架构设计 |
| 主频 | 每秒多少时钟周期 | CPU工艺、设计 |

### 三种周期的关系

$$
\text{总时钟周期数} = \text{指令数} \times \text{CPI}
$$

$$
T_{\text{CPU}} = \text{总时钟周期数} \times T_{\text{时钟}} = \frac{\text{总时钟周期数}}{f}
$$

## 原理分析

### 公式推导

从基本定义出发：

$$
T_{\text{CPU}} = \text{总时钟周期数} \times T_{\text{时钟}}
$$

$$
\text{总时钟周期数} = \sum_{i=1}^{n} (IC_i \times CPI_i)
$$

其中 $IC_i$ 是第 $i$ 类指令的数量，$CPI_i$ 是其CPI。

对于单一CPI的情况：
$$
T_{\text{CPU}} = IC \times CPI \times T_{\text{时钟}} = \frac{IC \times CPI}{f}
$$

### 各类指令CPI不同时的计算

当各类指令的CPI不同时：

$$
T_{\text{CPU}} = \frac{\sum_{i=1}^{n}(IC_i \times CPI_i)}{f} = \frac{IC_{\text{总}} \times \text{CPI}_{\text{avg}}}{f}
$$

$$
\text{CPI}_{\text{avg}} = \sum_{i=1}^{n} (CPI_i \times \frac{IC_i}{IC_{\text{总}}})
$$

## 直观理解

### 生活类比

CPU执行时间就像完成一项工程的总时间：

- **指令数** = 需要完成的任务数量（比如需要砌1000块砖）
- **CPI** = 每个任务需要的时间（比如每块砖需要2分钟）
- **主频** = 工人的工作速度（每分钟能做几个动作）

$$
\text{总时间} = \frac{\text{任务数} \times \text{每任务时间}}{\text{工作速度}}
$$

### 记忆技巧

CPU时间公式：「指令CPI除主频」
$$
T = \frac{IC \times CPI}{f}
$$

或者：「指令CPI乘周期」
$$
T = IC \times CPI \times T_{\text{时钟}}
$$

## 知识关联

### 与性能的关系

$$
P = \frac{1}{T_{\text{CPU}}} = \frac{f}{IC \times CPI}
$$

要提高性能：
1. **提高主频** $f$（但有功耗限制）
2. **减少指令数** $IC$（通过编译优化、更强大的指令集）
3. **降低CPI**（通过更好的流水线、Cache等）

### 与MIPS的关系

$$
\text{MIPS} = \frac{IC}{T_{\text{CPU}} \times 10^6} = \frac{f}{CPI \times 10^6}
$$

### 常考组合

1. 已知指令数、CPI、主频 → 求CPU执行时间
2. 已知CPU执行时间、CPI、主频 → 求指令数
3. 两台计算机的性能比较 → 通过CPU时间计算加速比

### 易错陷阱

1. **单位要统一**：主频用Hz，时间用s；或者主频用GHz，时间用ns
2. **CPI要用平均CPI**：当各类指令CPI不同时，要先算加权平均
3. **CPU执行时间不包括IO等待时间**：仅是CPU执行的时间
4. **指令数不等于程序中的指令条数**：循环会导致指令重复执行
