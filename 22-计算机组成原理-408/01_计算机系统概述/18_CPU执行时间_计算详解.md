# 18_CPU执行时间_计算详解

## 例题1：基本CPU时间计算

**题目**：某CPU主频为2.5GHz，执行某程序需要1.2亿条指令，平均CPI为1.6，求CPU执行时间。

**解答**：

$$
T_{\text{CPU}} = \frac{IC \times CPI}{f} = \frac{1.2 \times 10^8 \times 1.6}{2.5 \times 10^9}
$$

$$
= \frac{1.92 \times 10^8}{2.5 \times 10^9} = 0.0768\text{s} = 76.8\text{ms}
$$

## 例题2：各类指令CPI不同的情况

**题目**：某程序包含以下指令：
- A类指令：60000条，CPI = 1
- B类指令：30000条，CPI = 2
- C类指令：10000条，CPI = 4

CPU主频为1GHz。求（1）平均CPI；（2）CPU执行时间。

**解答**：

（1）平均CPI：

$$
\text{CPI}_{\text{avg}} = \frac{60000 \times 1 + 30000 \times 2 + 10000 \times 4}{60000 + 30000 + 10000}
$$

$$
= \frac{60000 + 60000 + 40000}{100000} = \frac{160000}{100000} = 1.6
$$

（2）CPU执行时间：

$$
T_{\text{CPU}} = \frac{100000 \times 1.6}{1 \times 10^9} = \frac{1.6 \times 10^5}{10^9} = 1.6 \times 10^{-4}\text{s} = 0.16\text{ms}
$$

## 例题3：两台计算机性能比较

**题目**：计算机A和B执行同一程序P，参数如下：

| 参数 | 计算机A | 计算机B |
|------|---------|---------|
| 主频 | 3GHz | 2.5GHz |
| 指令数 | 50亿 | 40亿 |
| 平均CPI | 1.2 | 1.5 |

（1）分别计算A和B执行程序P的时间。
（2）哪台计算机更快？快多少？

**解答**：

（1）执行时间：

$$
T_A = \frac{5 \times 10^9 \times 1.2}{3 \times 10^9} = \frac{6 \times 10^9}{3 \times 10^9} = 2.0\text{s}
$$

$$
T_B = \frac{4 \times 10^9 \times 1.5}{2.5 \times 10^9} = \frac{6 \times 10^9}{2.5 \times 10^9} = 2.4\text{s}
$$

（2）性能比较：

$$
\frac{T_B}{T_A} = \frac{2.4}{2.0} = 1.2
$$

A比B快20%。

## 例题4：优化指令比例后的性能变化

**题目**：某程序有三类指令，原始情况：

| 指令类型 | 比例 | CPI |
|---------|------|-----|
| 计算类 | 50% | 1 |
| 访存类 | 30% | 3 |
| 分支类 | 20% | 2 |

通过改进编译器，访存类比例降到15%，计算类比例升到65%。CPU主频为4GHz，程序有2亿条指令。

（1）求优化前后的平均CPI。
（2）求优化前后的CPU执行时间。
（3）性能提升了多少？

**解答**：

（1）平均CPI：

优化前：
$$
\text{CPI}_{\text{before}} = 1 \times 0.5 + 3 \times 0.3 + 2 \times 0.2 = 0.5 + 0.9 + 0.4 = 1.8
$$

优化后：
$$
\text{CPI}_{\text{after}} = 1 \times 0.65 + 3 \times 0.15 + 2 \times 0.2 = 0.65 + 0.45 + 0.4 = 1.5
$$

（2）CPU执行时间：

$$
T_{\text{before}} = \frac{2 \times 10^8 \times 1.8}{4 \times 10^9} = \frac{3.6 \times 10^8}{4 \times 10^9} = 0.09\text{s}
$$

$$
T_{\text{after}} = \frac{2 \times 10^8 \times 1.5}{4 \times 10^9} = \frac{3 \times 10^8}{4 \times 10^9} = 0.075\text{s}
$$

（3）性能提升：

$$
\text{加速比} = \frac{T_{\text{before}}}{T_{\text{after}}} = \frac{0.09}{0.075} = 1.2
$$

性能提升20%。

## 例题5：求未知的主频

**题目**：某程序有1.5亿条指令，平均CPI为2.0。要求CPU执行时间不超过0.1s，主频至少应为多少？

**解答**：

$$
T_{\text{CPU}} = \frac{IC \times CPI}{f} \leq 0.1
$$

$$
f \geq \frac{IC \times CPI}{T} = \frac{1.5 \times 10^8 \times 2.0}{0.1} = \frac{3 \times 10^8}{0.1} = 3 \times 10^9\text{Hz} = 3\text{GHz}
$$

主频至少应为3GHz。

## 计算技巧总结

### 常用数量级

$$
1\text{GHz} = 10^9\text{Hz}, \quad 1\text{MHz} = 10^6\text{Hz}
$$

$$
1\text{ms} = 10^{-3}\text{s}, \quad 1\mu\text{s} = 10^{-6}\text{s}, \quad 1\text{ns} = 10^{-9}\text{s}
$$

### 解题步骤

1. **明确已知量**：指令数、CPI、主频中的哪些已知
2. **统一单位**：主频转为Hz，时间转为s
3. **代入公式**：$T = \frac{IC \times CPI}{f}$
4. **检查答案**：单位是否合理，数量级是否正确
