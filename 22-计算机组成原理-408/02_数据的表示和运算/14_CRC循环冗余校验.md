# 15_CRC循环冗余校验

## 核心概念

### 定义
**CRC**（Cyclic Redundancy Check，循环冗余校验）是一种基于**多项式除法**的错误检测码。它通过在数据后面附加冗余位（余数），使得整个编码能被生成多项式整除。

### 核心要素
- **生成多项式**（Generator Polynomial）$G(x)$：预先约定的除数，是一个 $r$ 次多项式
- **数据多项式** $M(x)$：要发送的 $k$ 位数据对应的多项式
- **校验位**（CRC码）：数据后附加的 $r$ 位余数
- **模2运算**：加法 = 减法 = 异或（XOR），没有进位/借位

### 生成多项式的表示

生成多项式 $G(x)$ 的系数为0或1，例如：
- $G(x) = x^3 + x + 1$ 对应二进制 $1011$（$r = 3$）
- $G(x) = x^4 + x + 1$ 对应二进制 $10011$（$r = 4$）
- $G(x) = x^{16} + x^{15} + x^2 + 1$ 对应二进制 $11000000000000101$（$r = 16$）

### CRC编码过程
1. 设数据 $M$ 为 $k$ 位，生成多项式 $G(x)$ 为 $r+1$ 位
2. 在 $M$ 后面补 $r$ 个0，得到 $M'$
3. 用 $G(x)$ 对 $M'$ 进行**模2除法**（即异或运算），得到 $r$ 位余数 $R$
4. CRC编码 = $M$ 后面拼接 $R$

### 检错过程
1. 接收方收到编码后的 $k+r$ 位数据
2. 用同样的 $G(x)$ 对其进行模2除法
3. 如果余数为0：认为无错误
4. 如果余数不为0：检测到错误

---

## 原理分析

### 模2除法（核心运算）

**模2除法**与普通除法的区别：
- 每一步的减法用**异或**代替（没有借位）
- 商的每一位由当前被除数最高位决定：最高位为1则商1，为0则商0

### 编码示例

**题目**：数据 $M = 1101011$，生成多项式 $G(x) = x^3 + x + 1 = 1011$，求CRC码。

**Step 1**：$r = 3$，在数据后补3个0：$M' = 1101011000$

**Step 2**：模2除法 $1101011000 \div 1011$

```
              1110110    （商）
            ___________
  1011 ) 1101011000
         1011
         ----
          1100
          1011
          ----
           1111
           1011
           ----
            1000
            1011
            ----
             0110
             0000
             ----
              1100
              1011
              ----
               1110
               1011
               ----
                101    （余数）
```

**Step 3**：余数 $R = 101$

**CRC编码** = $1101011\mathbf{101}$

---

## 直观理解

### 生活类比
CRC就像给信件贴上一个"校验标签"：
- 发送方：把信件内容（数据）和约定的暗号（生成多项式）做运算，得到一个标签（余数）贴在信件末尾
- 接收方：用同样的暗号检查信件+标签，如果能"除尽"（余数=0），说明信件没被篡改

### 记忆技巧
1. **补零**：生成多项式有 $r+1$ 位，补 $r$ 个0
2. **模2除法**：减法就是异或，没有借位
3. **余数拼接**：余数 $r$ 位，拼接到数据末尾
4. **检错**：重新做模2除法，余数=0则无错

### CRC的优势
- 硬件实现简单（只需移位寄存器和异或门）
- 检错能力强（可检测所有≤r位的突发错误）
- 广泛用于网络（以太网、Wi-Fi）、磁盘存储、文件压缩等

---

## 知识关联

### 与海明码的对比

| 特性 | CRC | 海明码 |
|------|-----|--------|
| 设计目的 | 检错 | 检错+纠错 |
| 检错能力 | 可检测≤r位突发错误 | 可检测2位错误 |
| 纠错能力 | 无 | 纠正1位错误 |
| 运算方式 | 模2除法（多项式除法） | 奇偶校验（异或） |
| 硬件实现 | 移位寄存器+异或 | 组合逻辑 |
| 应用场景 | 数据链路层、存储 | 内存ECC |

### 常用生成多项式标准

| 标准 | 生成多项式 | 用途 |
|------|-----------|------|
| CRC-8 | $x^8 + x^2 + x + 1$ | 汽车电子 |
| CRC-16 | $x^{16} + x^{15} + x^2 + 1$ | Modbus通信 |
| CRC-32 | 多项式 | 以太网、ZIP |

## 代码/模拟

### Python实现CRC编码与检错

```python
"""CRC循环冗余校验模拟 - 适用于408考研复习"""

def xor_bits(a, b):
    """模2减法 = 异或（无借位）"""
    return [x ^ y for x, y in zip(a, b)]

def crc_encode(data_bits, generator_bits):
    """
    CRC编码
    :param data_bits: 数据位列表, 如 [1,1,0,1,0,1,1]
    :param generator_bits: 生成多项式系数, 如 [1,0,1,1] 对应 x^3+x+1
    :return: CRC编码 (数据 + 余数)
    """
    r = len(generator_bits) - 1  # 生成多项式的次数

    # Step 1: 数据末尾补r个0
    dividend = data_bits + [0] * r

    # Step 2: 模2除法
    working = dividend.copy()
    for i in range(len(data_bits)):
        if working[i] == 1:
            # 当前位为1, 做异或
            for j in range(len(generator_bits)):
                working[i + j] ^= generator_bits[j]

    # 余数在最后r位
    remainder = working[-r:]
    codeword = data_bits + remainder

    return codeword, remainder

def crc_check(received_bits, generator_bits):
    """CRC检错: 重新做模2除法, 余数为0则无错"""
    r = len(generator_bits) - 1
    working = received_bits.copy()

    for i in range(len(received_bits) - r):
        if working[i] == 1:
            for j in range(len(generator_bits)):
                working[i + j] ^= generator_bits[j]

    remainder = working[-r:]
    return all(b == 0 for b in remainder), remainder

# 笔记中的示例: 数据1101011, 生成多项式1011 (x^3+x+1)
print("=== CRC编码示例 ===")
data = [1, 1, 0, 1, 0, 1, 1]
generator = [1, 0, 1, 1]  # x^3 + x + 1

print(f"数据: {''.join(str(b) for b in data)}")
print(f"生成多项式: {''.join(str(b) for b in generator)} (x^3+x+1)")

# 编码
codeword, remainder = crc_encode(data, generator)
print(f"余数: {''.join(str(b) for b in remainder)}")
print(f"CRC编码: {''.join(str(b) for b in codeword)}")

# 检错验证
valid, rem = crc_check(codeword, generator)
print(f"\n验证(无错误): 余数={''.join(str(b) for b in rem)}, "
      f"{'无错 ✓' if valid else '有错 ✗'}")

# 模拟1位错误
corrupted = codeword.copy()
corrupted[3] = 1 - corrupted[3]  # 翻转一位
valid2, rem2 = crc_check(corrupted, generator)
print(f"验证(1位错误): 余数={''.join(str(b) for b in rem2)}, "
      f"{'无错 ✓' if valid2 else '有错 ✗'}")

# 模拟2位错误
corrupted2 = codeword.copy()
corrupted2[2] = 1 - corrupted2[2]
corrupted2[5] = 1 - corrupted2[5]
valid3, rem3 = crc_check(corrupted2, generator)
print(f"验证(2位错误): 余数={''.join(str(b) for b in rem3)}, "
      f"{'无错 ✓' if valid3 else '有错 ✗'}")
```

### 手动模2除法过程演示

```python
def crc_step_by_step(data, generator):
    """逐步展示模2除法过程"""
    r = len(generator) - 1
    dividend = data + [0] * r
    working = dividend.copy()

    print(f"数据: {''.join(str(b) for b in data)}, 补{r}个0")
    print(f"生成多项式: {''.join(str(b) for b in generator)}")
    print(f"被除数: {''.join(str(b) for b in dividend)}\n")

    for i in range(len(data)):
        if working[i] == 1:
            old = working[i:i+len(generator)]
            result = xor_bits(old, generator)
            working[i:i+len(generator)] = result
            print(f"第{i+1}步: 商1, 异或{''.join(str(b) for b in generator)}"
                  f" → {''.join(str(b) for b in working)}")
        else:
            print(f"第{i+1}步: 商0, 不做操作")

    remainder = working[-r:]
    print(f"\n余数 = {''.join(str(b) for b in remainder)}")
    return data + remainder

print("\n=== 手动模2除法过程 ===")
crc_step_by_step([1,1,0,1,0,1,1], [1,0,1,1])
```

### 408考试要点
- 模2除法的正确执行（最核心技能）
- 补零数量 = 生成多项式次数（最高次幂）
- 余数位数 = 生成多项式次数
- CRC编码 = 数据 + 余数
