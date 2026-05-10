# 位运算专题 (Bit Manipulation)

## 一、基础操作

### 1.1 六种基本运算

| 运算符 | 名称 | 示例 | 说明 |
|--------|------|------|------|
| `&` | 按位与 | `5 & 3 = 1` | 两位都为1时结果为1 |
| `\|` | 按位或 | `5 \| 3 = 7` | 至少一位为1时结果为1 |
| `^` | 按位异或 | `5 ^ 3 = 6` | 两位不同时结果为1 |
| `~` | 按位取反 | `~5 = -6` | 0变1，1变0 |
| `<<` | 左移 | `5 << 1 = 10` | 乘以2 |
| `>>` | 右移 | `5 >> 1 = 2` | 除以2 |

### 1.2 常用技巧

```python
# 判断奇偶
n & 1  # 1为奇数，0为偶数

# 交换两数（不用额外变量）
a ^= b; b ^= a; a ^= b

# 取最低位的1
lowbit = n & (-n)

# 去掉最低位的1
n & (n - 1)

# 判断是否为2的幂
n > 0 and (n & (n - 1)) == 0

# 枚举子集
sub = mask
while sub:
    # 处理 sub
    sub = (sub - 1) & mask
```

---

## 二、经典题目详解

### 2.1 只出现一次的数字 (LeetCode 136)

**异或性质：** `a ^ a = 0`, `a ^ 0 = a`

```python
def single_number(nums):
    result = 0
    for num in nums:
        result ^= num
    return result
```

### 2.2 只出现一次的数字II (LeetCode 137)

每个数出现3次，用状态机。

```python
def single_number_ii(nums):
    ones = twos = 0
    for num in nums:
        ones = (ones ^ num) & ~twos
        twos = (twos ^ num) & ~ones
    return ones
```

### 2.3 只出现一次的数字III (LeetCode 260)

两个数各出现一次，其余出现两次。

```python
def single_number_iii(nums):
    xor = 0
    for num in nums:
        xor ^= num
    # 找到最低位的1来区分两组
    diff = xor & (-xor)
    a = b = 0
    for num in nums:
        if num & diff: a ^= num
        else: b ^= num
    return [a, b]
```

### 2.4 位1的个数 (LeetCode 191)

```python
def hamming_weight(n):
    count = 0
    while n:
        n &= n - 1  # 去掉最低位的1
        count += 1
    return count
```

### 2.5 2的幂 (LeetCode 231)

```python
def is_power_of_two(n):
    return n > 0 and (n & (n - 1)) == 0
```

### 2.6 比特位计数 (LeetCode 338)

```python
def count_bits(n):
    dp = [0] * (n + 1)
    for i in range(1, n + 1):
        dp[i] = dp[i >> 1] + (i & 1)
    return dp
```

### 2.7 最大单词长度乘积 (LeetCode 318)

用位掩码表示每个单词的字符集合。

```python
def max_product(words):
    masks = []
    for word in words:
        mask = 0
        for c in word:
            mask |= 1 << (ord(c) - ord('a'))
        masks.append(mask)

    result = 0
    for i in range(len(words)):
        for j in range(i+1, len(words)):
            if masks[i] & masks[j] == 0:
                result = max(result, len(words[i]) * len(words[j]))
    return result
```

---

## 三、状态压缩

### 3.1 用整数表示集合

```python
# 表示集合 {0, 2, 3} → 0b1101 = 13
mask = (1 << 0) | (1 << 2) | (1 << 3)

# 检查元素是否在集合中
if mask & (1 << i):  # i 在集合中

# 添加/删除元素
mask |= (1 << i)   # 添加
mask &= ~(1 << i)  # 删除

# 枚举所有子集
sub = mask
while sub:
    # 处理 sub
    sub = (sub - 1) & mask
```

### 3.2 子集枚举 (LeetCode 78)

```python
def subsets(nums):
    n = len(nums)
    result = []
    for mask in range(1 << n):
        subset = []
        for i in range(n):
            if mask & (1 << i):
                subset.append(nums[i])
        result.append(subset)
    return result
```

---

## 四、C++ 实现

```cpp
#include <bits/stdc++.h>
using namespace std;

// 只出现一次的数字
int singleNumber(vector<int>& nums) {
    int result = 0;
    for (int num : nums) result ^= num;
    return result;
}

// 位1的个数
int hammingWeight(uint32_t n) {
    int count = 0;
    while (n) { n &= n - 1; count++; }
    return count;
}

// 比特位计数
vector<int> countBits(int n) {
    vector<int> dp(n + 1, 0);
    for (int i = 1; i <= n; i++)
        dp[i] = dp[i >> 1] + (i & 1);
    return dp;
}
```

---

## 五、复杂度分析

| 问题 | 时间复杂度 | 空间复杂度 |
|------|-----------|-----------|
| 单独数字 | $O(n)$ | $O(1)$ |
| 汉明重量 | $O(k)$ k为1的个数 | $O(1)$ |
| 比特位计数 | $O(n)$ | $O(n)$ |
| 子集枚举 | $O(n \cdot 2^n)$ | $O(n)$ |
| 状态压缩DP | $O(2^n \cdot n)$ | $O(2^n)$ |

---

## 六、面试高频题

1. **LeetCode 136：** 只出现一次的数字
2. **LeetCode 191：** 位1的个数
3. **LeetCode 231：** 2的幂
4. **LeetCode 338：** 比特位计数
5. **LeetCode 260：** 只出现一次的数字III
6. **LeetCode 318：** 最大单词长度乘积
7. **LeetCode 371：** 两整数之和
8. **LeetCode 461：** 汉明距离
9. **LeetCode 78：** 子集（位掩码法）
10. **LeetCode 187：** 重复的DNA序列
