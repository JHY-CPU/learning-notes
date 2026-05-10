# 位运算实现加法 (Bitwise Addition)

## 一、原理

### 1.1 加法的位运算分解

二进制加法可以分解为两部分：
1. **无进位和：** 对应异或运算 `a ^ b`
2. **进位：** 对应与运算左移 `(a & b) << 1`

不断将进位加到无进位和上，直到没有进位为止。

### 1.2 示例

计算 `5 + 3`（二进制 `101 + 011`）：

```
第1轮: a=101, b=011
  无进位和: 101^011 = 110
  进位:     101&011 = 001, 左移 = 010

第2轮: a=110, b=010
  无进位和: 110^010 = 100
  进位:     110&010 = 010, 左移 = 100

第3轮: a=100, b=100
  无进位和: 100^100 = 000
  进位:     100&100 = 100, 左移 = 1000

第4轮: a=000, b=1000
  无进位和: 000^1000 = 1000
  进位:     000&1000 = 0000

进位为0，结束。结果: 1000 = 8 ✓
```

---

## 二、代码实现

### 2.1 Python 实现

```python
def add(a, b):
    """用位运算实现加法"""
    # Python整数无限精度，需要模拟32位
    MASK = 0xFFFFFFFF  # 32位掩码
    MAX_INT = 0x7FFFFFFF

    a, b = a & MASK, b & MASK
    while b != 0:
        carry = (a & b) << 1
        a = (a ^ b) & MASK
        b = carry & MASK

    # 处理负数
    return a if a <= MAX_INT else ~(a ^ MASK)

print(add(5, 3))    # 8
print(add(-5, 3))   # -2
print(add(-5, -3))  # -8
```

### 2.2 C++ 实现

```cpp
#include <bits/stdc++.h>
using namespace std;

int add(int a, int b) {
    while (b != 0) {
        int carry = (unsigned)(a & b) << 1;  // 进位
        a = a ^ b;                           // 无进位和
        b = carry;
    }
    return a;
}

int main() {
    cout << add(5, 3) << endl;     // 8
    cout << add(-5, 3) << endl;    // -2
    cout << add(-5, -3) << endl;   // -8
    return 0;
}
```

### 2.3 递归版本

```python
def add_recursive(a, b):
    if b == 0:
        return a
    return add_recursive(a ^ b, (a & b) << 1)
```

---

## 三、用加法实现减法和乘法

### 3.1 减法

减法等于加上相反数：`a - b = a + (-b)`

在补码表示中，`-b = ~b + 1`

```python
def subtract(a, b):
    return add(a, add(~b, 1))

print(subtract(5, 3))   # 2
print(subtract(3, 5))   # -2
```

### 3.2 乘法（俄罗斯农民法）

```python
def multiply(a, b):
    result = 0
    # 处理负数
    negative = False
    if b < 0:
        b = add(~b, 1)  # 取反加1 = 取相反数
        negative = True

    while b != 0:
        if b & 1:  # b的最低位为1
            result = add(result, a)
        a = a << 1   # a翻倍
        b = b >> 1   # b除以2

    if negative:
        result = add(~result, 1)  # 取相反数

    return result

print(multiply(3, 5))   # 15
print(multiply(-3, 5))  # -15
```

### 3.3 除法（二分+减法）

```python
def divide(dividend, divisor):
    # 处理特殊情况
    if dividend == -2147483648 and divisor == -1:
        return 2147483647  # 溢出

    negative = (dividend < 0) ^ (divisor < 0)
    dvd = abs(dividend)
    dvs = abs(divisor)
    quotient = 0

    while dvd >= dvs:
        shift = 0
        while dvd >= (dvs << shift):
            shift += 1
        shift -= 1
        quotient += (1 << shift)
        dvd -= (dvs << shift)

    return -quotient if negative else quotient
```

---

## 四、LeetCode 371 — 两整数之和

**问题：** 不使用 `+` 和 `-` 运算符，计算两整数之和。

```python
def get_sum(a, b):
    MASK = 0xFFFFFFFF
    MAX_INT = 0x7FFFFFFF

    a, b = a & MASK, b & MASK
    while b != 0:
        carry = (a & b) << 1
        a = (a ^ b) & MASK
        b = carry & MASK

    return a if a <= MAX_INT else ~(a ^ MASK)
```

---

## 五、复杂度分析

| 操作 | 时间复杂度 | 说明 |
|------|-----------|------|
| 加法 | $O(\log n)$ | 最多循环32次（32位整数） |
| 减法 | $O(\log n)$ | 调用两次加法 |
| 乘法 | $O(\log n)$ | 最多循环32次 |
| 除法 | $O(\log^2 n)$ | 二分搜索 |

---

## 六、面试要点

1. **理解进位机制** — 与运算+左移
2. **Python的特殊处理** — 无限精度需要模拟32位
3. **负数处理** — 补码表示
4. **扩展到减法乘法** — 展示系统性思维
