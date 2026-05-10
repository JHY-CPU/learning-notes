# Sprague-Grundy 函数 (SG函数)

## 一、概念定义与原理

### 1.1 背景

Nim 定理只能处理简单的取石子游戏。**Sprague-Grundy 定理**将任意公平组合游戏转化为等价的 Nim 游戏。

### 1.2 SG 函数定义

对于博弈状态 $x$，定义其 SG 值为：

$$SG(x) = \text{mex}(\{SG(y) \mid x \to y\})$$

其中 $\text{mex}(S) = \min\{n \geq 0 \mid n \notin S\}$（最小非负整数不在集合 $S$ 中）。

**终态：** $SG(\text{终态}) = 0$（无法操作，后继集合为空，$\text{mex}(\emptyset) = 0$）

### 1.3 Sprague-Grundy 定理

**定理：** 多个独立子游戏的复合博弈，其 SG 值为各子游戏 SG 值的异或：

$$SG(\text{复合状态}) = SG(\text{子游戏}_1) \oplus SG(\text{子游戏}_2) \oplus \cdots \oplus SG(\text{子游戏}_k)$$

- SG 值非零：先手必胜（N-position）
- SG 值为零：先手必败（P-position）

---

## 二、核心算法

### 2.1 SG 函数计算

**记忆化搜索：** 对每个状态 $x$，递归计算所有后继状态的 SG 值，然后取 mex。

**直接递推：** 如果状态有明确的递推关系，可以从小到大计算。

### 2.2 经典例子

**取石子（每次取 $1 \sim m$ 个）：** $SG(x) = x \bmod (m+1)$

**取石子（每次取 $2^k$ 个）：** $SG(x) = x \bmod 3$

---

## 三、代码实现

### 3.1 SG 函数记忆化搜索 - C++

```cpp
#include <bits/stdc++.h>
using namespace std;

map<int, int> sg_cache;

int sg(int x, vector<int>& moves) {
    if (sg_cache.count(x)) return sg_cache[x];
    set<int> reachable;
    for (int m : moves) {
        if (x - m >= 0) reachable.insert(sg(x - m, moves));
    }
    // 求 mex
    int mex = 0;
    while (reachable.count(mex)) mex++;
    return sg_cache[x] = mex;
}

// 判断复合博弈的胜负
bool sprague_grundy(vector<int>& states, vector<int>& moves) {
    int xor_sum = 0;
    for (int s : states) {
        sg_cache.clear();
        xor_sum ^= sg(s, moves);
    }
    return xor_sum != 0;
}
```

### 3.2 递推计算 SG 值 - C++

```cpp
const int MAXN = 10005;
int sg_val[MAXN];

void compute_sg(int n, vector<int>& moves) {
    for (int i = 0; i <= n; i++) {
        set<int> reachable;
        for (int m : moves) {
            if (i - m >= 0) reachable.insert(sg_val[i - m]);
        }
        int mex = 0;
        while (reachable.count(mex)) mex++;
        sg_val[i] = mex;
    }
}
```

### 3.3 Python 实现

```python
def sg(x, moves, cache={}):
    """计算状态x的SG值"""
    if x in cache: return cache[x]
    reachable = set()
    for m in moves:
        if x - m >= 0:
            reachable.add(sg(x - m, moves, cache))
    mex = 0
    while mex in reachable: mex += 1
    cache[x] = mex
    return mex

def sprague_grundy(states, moves):
    """判断复合博弈胜负"""
    xor_sum = 0
    for s in states:
        xor_sum ^= sg(s, moves)
    return xor_sum != 0

# 测试：每次取1~3个石子
moves = [1, 2, 3]
for i in range(10):
    print(f"SG({i}) = {sg(i, moves, {})}")
# SG值: 0,1,2,3,0,1,2,3,0,1 (周期为4)
```

### 3.4 实际竞赛应用

```cpp
// 求阶梯 Nim 的 SG 值
// 把奇数编号的堆做 Nim 即可
bool stair_nim(vector<int>& piles) {
    int xor_sum = 0;
    for (int i = 1; i < piles.size(); i += 2) { // 奇数堆（1-indexed）
        xor_sum ^= piles[i];
    }
    return xor_sum != 0;
}
```

---

## 四、复杂度分析

| 操作 | 时间复杂度 | 说明 |
|------|-----------|------|
| 单个SG值计算 | $O(\text{状态数} \times \text{分支数})$ | 记忆化 |
| 批量SG值 | $O(n \times k)$ | $n$ 为状态数，$k$ 为分支数 |
| 复合博弈判断 | $O(\sum \text{各子游戏复杂度})$ | 异或合并 |

---

## 五、竞赛与面试应用场景

### 5.1 解题框架

1. **分解子游戏：** 将复合博弈分解为独立子游戏
2. **计算SG值：** 对每个子游戏计算 SG 函数
3. **异或判断：** 将所有 SG 值异或，非零先手必胜

### 5.2 常见变种

- **阶梯 Nim：** 只有奇数堆有效
- **翻硬币游戏：** 每次翻转一枚或连续几枚
- **图上博弈：** 在 DAG 上定义博弈

### 5.3 注意事项

- SG 函数是博弈论的通用工具
- 记忆化是必须的，避免重复计算
- 注意状态的定义，确保无环
