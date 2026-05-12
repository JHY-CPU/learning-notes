# 50_Booth算法

## 核心概念

### Booth算法的定义

Booth算法是一种高效的**补码乘法算法**，由Andrew Booth于1951年提出。它通过检查乘数的最低两位来决定每轮的操作，实现了补码乘法的自动化。

### Booth算法的核心思想

每轮检查乘数的**最低两位** $(y_i, y_{i+1})$，根据其值决定操作：

| $(y_i, y_{i+1})$ | 操作 | 数学含义 |
|-------------------|------|---------|
| $00$ | 只右移，不加 | 该位范围全为0 |
| $01$ | 加 $[x]_补$，再右移 | 该位有一个1 |
| $10$ | 加 $[-x]_补$，再右移 | 该位范围有0到1的跳变 |
| $11$ | 只右移，不加 | 该位范围全为1 |

### 关键要素

**需要准备的数据**：
- $[x]_补$：被乘数的补码
- $[-x]_补$：被乘数的补码取负（连同符号位取反加1）
- $[y]_补$：乘数的补码
- 附加位 $y_{n+1} = 0$

**寄存器设置**：
- ACC：存放部分积（n+1 位），初始为 0
- MQ：存放乘数 $[y]_补$（n 位），右移过程中逐步移出
- 附加位 $y_{n+1}$：1 位，初始为 0
- X：存放 $[x]_补$
- $\bar{X}$：存放 $[-x]_补$

### Booth算法的完整流程

```
初始化：
  ACC ← 0...0 (n+1位)
  MQ ← [y]_补 (n位)
  y_{n+1} ← 0
  计数器 C ← n

循环（共n轮）：
  1. 检查 (MQ最低位, y_{n+1}) = (y_n, y_{n+1})
  2. 根据判断结果执行加法：
     - 00 或 11：不加
     - 01：ACC ← ACC + [x]_补
     - 10：ACC ← ACC + [-x]_补
  3. 算术右移：(ACC, MQ, y_{n+1}) 联合右移一位
     - ACC 高位补符号位
     - y_{n+1} 接收 MQ 最低位
  4. C ← C - 1，若 C ≠ 0 回到步骤1

结束：
  结果在 ACC（高位）和 MQ（低位）中，共 2n 位补码
```

**注意**：最后一步右移后结束。不同教材对"最后一步是否右移"的表述可能不同，以考试指定教材为准。唐朔飞版通常 n 轮中每轮都移位；白中英版可能最后一步不移位。

### Booth算法的本质理解

Booth算法的本质是利用相邻两位的差值来表示乘数：

$$y = \sum_{i=0}^{n-1} (y_i - y_{i+1}) \times 2^{i}$$

其中 $y_n$（附加位）= 0。

- $(y_i, y_{i+1}) = (0, 0)$：差值为 0，不加
- $(y_i, y_{i+1}) = (0, 1)$：差值为 +1，加一个 $[x]_补$
- $(y_i, y_{i+1}) = (1, 0)$：差值为 -1，加一个 $[-x]_补$
- $(y_i, y_{i+1}) = (1, 1)$：差值为 0，不加

## 原理分析

### 为什么检查最低两位有效？

以 4 位乘数为例，设乘数 $y = 0101$（即 +5）：

用 Booth 编码（从低位到高位，含附加位0）：
- $(y_4, y_5) = (1, 0)$：加 $[-x]_补$
- $(y_3, y_4) = (0, 1)$：加 $[x]_补$
- $(y_2, y_3) = (1, 0)$：加 $[-x]_补$
- $(y_1, y_2) = (0, 1)$：加 $[x]_补$

这等价于：$x \times (2^3 - 2^2 + 2^1 - 2^0) \times 2^{-4}$...（经过移位修正后等于 $x \times 5$）

实际上，Booth编码将连续的1串合并为减法和加法，减少了加法次数。

### 硬件实现

Booth算法的硬件需要：
- 一个加法器
- 三个寄存器（ACC, MQ, X/NOTX）
- 一个控制逻辑（检查最低两位，选择加什么）
- 移位网络

```
      ┌──────────┐
      │   ACC    │←── 部分积
      └────┬─────┘
           │ 加法结果
           ▼
      ┌──────────┐     ┌──────────┐
      │  加法器  │←────│    X     │  [x]_补
      │          │←────│   ~X+1   │  [-x]_补
      └──────────┘     └──────────┘
           ▲
      ┌────┴─────┐
      │   MQ     │───→ 检查最低两位
      └──────────┘
      ┌──────────┐
      │ 附加位y  │
      └──────────┘
```

## 直观理解

### 生活类比

Booth算法类似于"合并同类项"：
- 连续一串 1（如 011110）可以写成 $100000 - 00010$（高位减低位）
- 这样一次减法就替代了多次加法

### 记忆口诀

- **"00、11只右移，01加正10加负"**
- **"附加位初始为0"**
- **"算术右移，高位补符号"**

### 与原码乘法的效率对比

| 乘数模式 | 原码一位乘加法次数 | Booth算法加法次数 |
|---------|------------------|-----------------|
| 01010101 | 4次 | 最多4次 |
| 11111111 | 8次 | 2次（10-01）|
| 10000000 | 1次 | 2次 |

Booth算法在乘数包含长串 1 时效率更高。

## 知识关联

- 补码一位乘（第49节）是Booth算法的理论基础
- 原码一位乘（第46节）是Booth算法的对比对象
## 代码/模拟

### Python实现Booth算法

```python
"""Booth补码乘法算法模拟 - 适用于408考研复习"""

def booth_multiply(x, y, n_bits=8):
    """
    Booth算法实现补码乘法
    :param x: 被乘数（有符号整数）
    :param y: 乘数（有符号整数）
    :param n_bits: 操作位数
    :return: 2n位补码乘积
    """
    def to_twos_comp(val, bits):
        """整数转补码（无符号表示）"""
        if val < 0:
            val = (1 << bits) + val
        return val & ((1 << bits) - 1)

    def from_twos_comp(val, bits):
        """补码转有符号整数"""
        if val & (1 << (bits - 1)):
            val -= (1 << bits)
        return val

    def arithmetic_right_shift(acc, mq, y_extra, bits):
        """算术右移: (ACC, MQ, y_{n+1}) 联合右移一位"""
        # 新的y_{n+1} = MQ最低位
        new_y_extra = mq & 1
        # MQ右移：高位由ACC最低位补入
        new_mq = ((acc & 1) << (bits - 1)) | (mq >> 1)
        # ACC算术右移：高位补符号位
        sign = acc & (1 << bits)  # 保留符号位
        new_acc = (acc >> 1) & ((1 << bits) - 1)
        if sign:
            new_acc |= (1 << (bits - 1))  # 补符号位
        return new_acc, new_mq, new_y_extra

    mask = (1 << n_bits) - 1

    # 初始化
    acc = 0                            # ACC = 0 (n位)
    mq = to_twos_comp(y, n_bits)       # MQ = 乘数补码
    x_comp = to_twos_comp(x, n_bits)   # [x]补
    neg_x_comp = to_twos_comp(-x, n_bits)  # [-x]补
    y_extra = 0                        # 附加位 y_{n+1} = 0

    print(f"被乘数 x={x}, [x]补 = {x_comp:0{n_bits}b} = 0x{x_comp:02X}")
    print(f"乘数   y={y}, [y]补 = {mq:0{n_bits}b} = 0x{mq:02X}")
    print(f"[-x]补 = {neg_x_comp:0{n_bits}b} = 0x{neg_x_comp:02X}")
    print(f"\n{'步骤':<4} {'(y0,y1)':<10} {'操作':<20} {'ACC':>10} {'MQ':>10} {'y+1':>4}")
    print("-" * 62)

    for step in range(n_bits):
        y0 = mq & 1  # MQ最低位
        pair = (y0, y_extra)

        # 判断操作
        if pair == (0, 0) or pair == (1, 1):
            operation = "不加（只右移）"
            acc = acc  # 不变
        elif pair == (0, 1):
            operation = f"ACC + [x]补"
            acc = (acc + x_comp) & mask
        elif pair == (1, 0):
            operation = f"ACC + [-x]补"
            acc = (acc + neg_x_comp) & mask

        print(f" {step+1:<3} ({y0},{y_extra})     {operation:<20} "
              f"{acc:0{n_bits}b}   {mq:0{n_bits}b}   {y_extra}")

        # 算术右移
        acc, mq, y_extra = arithmetic_right_shift(acc, mq, y_extra, n_bits)

    # 最终结果：ACC(高位) MQ(低位) = 2n位补码
    result = (acc << n_bits) | mq
    expected = x * y
    print(f"\n结果: ACC={acc:0{n_bits}b}, MQ={mq:0{n_bits}b}")
    print(f"乘积 = {acc:0{n_bits}b}_{mq:0{n_bits}b} = {from_twos_comp(result, 2*n_bits)}")
    print(f"验证: {x} × {y} = {expected} ✓" if from_twos_comp(result, 2*n_bits) == expected
          else f"验证失败: 期望{expected}")
    return result

# 示例：与笔记中一致的计算
print("=== Booth算法示例: (-3) × (+5) ===")
booth_multiply(-3, 5, n_bits=4)

print("\n=== Booth算法示例: (+7) × (-2) ===")
booth_multiply(7, -2, n_bits=4)
```

### Booth编码优化说明

```python
def booth_encoding_demo(y, n_bits=8):
    """展示Booth编码如何减少连续1串的加法次数"""
    y_bin = format(y & ((1 << n_bits) - 1), f'0{n_bits}b')
    print(f"\n乘数 y={y}, 二进制: {y_bin}0 (附加位0)")

    print("Booth编码 (从低位到高位, 检查相邻两位):")
    print(f"  位对:   ", end='')
    pairs = []
    extended = y_bin + '0'  # 加上附加位
    for i in range(n_bits):
        pair = (int(extended[i+1]), int(extended[i]))
        pairs.append(pair)
        print(f"({extended[i+1]},{extended[i]}) ", end='')
    print()

    print(f"  操作:   ", end='')
    for pair in pairs:
        if pair == (0, 0) or pair == (1, 1):
            print(f"  0   ", end='')
        elif pair == (0, 1):
            print(f" +x   ", end='')
        elif pair == (1, 0):
            print(f" -x   ", end='')
    print()

booth_encoding_demo(0b01010101)  # 常规模式
booth_encoding_demo(0b11111111)  # 全1串 - Booth编码只需2次操作!
booth_encoding_demo(0b10000000)  # 单个1
```

- Booth算法是408考试的**高频考点**，常以大题形式出现
- 算术右移与逻辑右移的区别（第1章已学）
