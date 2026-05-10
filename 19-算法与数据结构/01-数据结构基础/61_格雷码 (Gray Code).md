# 格雷码 (Gray Code)

## 一、概念定义

### 1.1 什么是格雷码

格雷码 (Gray Code) 是一种**相邻两个数的二进制表示只有一位不同**的编码方式。

**n位格雷码：** 共 $2^n$ 个码字，从 $0$ 到 $2^n - 1$ 的排列，相邻码字仅一位不同。

### 1.2 示例

**3位格雷码：**

| 十进制 | 二进制 | 格雷码 |
|--------|--------|--------|
| 0 | 000 | 000 |
| 1 | 001 | 001 |
| 2 | 010 | 011 |
| 3 | 011 | 010 |
| 4 | 100 | 110 |
| 5 | 101 | 111 |
| 6 | 110 | 101 |
| 7 | 111 | 100 |

---

## 二、生成方法

### 2.1 公式法

格雷码可以通过以下公式从二进制数生成：

$$G(i) = i \oplus (i >> 1)$$

```python
def gray_code(n):
    return [i ^ (i >> 1) for i in range(1 << n)]

print(gray_code(3))
# [0, 1, 3, 2, 6, 7, 5, 4]
```

**C++ 实现：**

```cpp
#include <bits/stdc++.h>
using namespace std;

vector<int> grayCode(int n) {
    vector<int> result(1 << n);
    for (int i = 0; i < (1 << n); i++) {
        result[i] = i ^ (i >> 1);
    }
    return result;
}
```

### 2.2 镜像法（递归构造）

**思路：** n位格雷码可以通过 n-1 位格雷码构造：
1. 在 n-1 位格雷码前加 0（正序）
2. 在 n-1 位格雷码前加 1（逆序）
3. 拼接

```python
def gray_code_mirror(n):
    if n == 0:
        return [0]

    prev = gray_code_mirror(n - 1)
    # 前半部分：在前面加0
    result = prev[:]
    # 后半部分：在前面加1（逆序）
    for i in range(len(prev) - 1, -1, -1):
        result.append(prev[i] | (1 << (n - 1)))

    return result

print(gray_code_mirror(3))
# [0, 1, 3, 2, 6, 7, 5, 4]
```

**图解 n=3 的构造过程：**

```
n=1: 0, 1

n=2:
  前加0: 00, 01
  前加1: 11, 10
  合并:  00, 01, 11, 10

n=3:
  前加0: 000, 001, 011, 010
  前加1: 110, 111, 101, 100
  合并:  000, 001, 011, 010, 110, 111, 101, 100
```

### 2.3 从格雷码还原二进制

```python
def gray_to_binary(gray):
    binary = gray
    while gray >>= 1:  # Python不支持这样写，仅示意
        binary ^= gray
    return binary

# Python正确实现
def gray_to_binary(g):
    b = g
    shift = 1
    while (g >> shift) > 0:
        b ^= (g >> shift)
        shift += 1
    return b
```

---

## 三、LeetCode 89 — 格雷码

**问题：** 给定n，返回n位格雷码序列。要求首尾也只差一位（循环格雷码）。

### 3.1 解法

```python
def gray_code(n):
    # 公式法
    return [i ^ (i >> 1) for i in range(1 << n)]
```

### 3.2 验证循环条件

```python
def is_valid_gray_code(seq):
    n = len(seq)
    for i in range(n):
        diff = seq[i] ^ seq[(i + 1) % n]
        # 检查是否只有一位不同
        if bin(diff).count('1') != 1:
            return False
    return True
```

---

## 四、应用场景

### 4.1 数字电路

格雷码在数字电路中用于减少切换错误。相邻数字只变一位，减少电磁干扰。

### 4.2 旋转编码器

机械旋转编码器使用格雷码，避免同时多位变化导致的读数错误。

### 4.3 状态压缩DP

在一些状态压缩问题中，如果需要枚举所有状态且状态转移只变一位，格雷码顺序可以优化。

---

## 五、复杂度分析

| 操作 | 时间 | 空间 |
|------|------|------|
| 生成格雷码 | $O(2^n)$ | $O(2^n)$ |
| 公式法单个 | $O(1)$ | $O(1)$ |
| 镜像法 | $O(2^n)$ | $O(2^n)$ |
| 格雷码转二进制 | $O(n)$ | $O(1)$ |

---

## 六、面试要点

1. **公式 $G(i) = i \oplus (i >> 1)$** — 必须记住
2. **镜像构造法** — 理解递归思想
3. **循环格雷码** — 首尾也只差一位
4. **应用了解** — 电路、编码器、DP优化
