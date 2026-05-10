# 博弈论基础 (Game Theory)

## 一、概念定义与原理

### 1.1 组合博弈基本概念

- **公平组合游戏（ICG）：** 两名玩家轮流操作，操作规则相同，无法操作者输
- **P-position（必败态）：** 前一个玩家（Previous）必胜，即当前玩家必败
- **N-position（必胜态）：** 下一个玩家（Next）必胜

### 1.2 基本定理

**定理：**
- 终态是 P-position
- 可以移动到 P-position 的状态是 N-position
- 所有后继都是 N-position 的状态是 P-position

---

## 二、核心算法

### 2.1 Nim 游戏

有 $n$ 堆石子，每堆有 $a_i$ 个。两人轮流取，每次从一堆中取至少一个。

**定理（Bouton定理）：** 当且仅当 $a_1 \oplus a_2 \oplus \cdots \oplus a_n \neq 0$ 时，先手必胜。

**证明：** 异或和为0时，任何操作都会使异或和非0；异或和非0时，总可以找到一种操作使其变为0。

### 2.2 威佐夫博弈

两堆石子 $(a, b)$，每次可以从一堆取任意个，或从两堆取相同个。

**结论：** 先手必败当且仅当：

$$\min(a, b) = \left\lfloor \frac{\sqrt{5} + 1}{2} \cdot |a - b| \right\rfloor$$

### 2.3 巴什博弈

$n$ 个石子，每次取 $1 \sim m$ 个。

**结论：** 先手必败当且仅当 $n \bmod (m+1) = 0$。

---

## 三、代码实现

### 3.1 Nim 游戏 - C++

```cpp
#include <bits/stdc++.h>
using namespace std;

// Nim 游戏判断先手胜负
bool nim_game(vector<int>& piles) {
    int xor_sum = 0;
    for (int p : piles) xor_sum ^= p;
    return xor_sum != 0; // true 表示先手必胜
}

// 找到 Nim 游戏的必胜策略
pair<int, int> nim_move(vector<int>& piles) {
    int xor_sum = 0;
    for (int p : piles) xor_sum ^= p;
    if (xor_sum == 0) return {-1, -1}; // 必败
    for (int i = 0; i < piles.size(); i++) {
        int target = piles[i] ^ xor_sum;
        if (target < piles[i]) {
            return {i, piles[i] - target}; // 从第 i 堆取 piles[i]-target 个
        }
    }
    return {-1, -1};
}
```

### 3.2 威佐夫博弈 - C++

```cpp
bool wythoff_game(int a, int b) {
    if (a > b) swap(a, b);
    double golden = (sqrt(5.0) + 1) / 2;
    return a != (int)(golden * (b - a));
    // true 表示先手必胜
}
```

### 3.3 Python 实现

```python
import math

def nim_game(piles):
    """Nim游戏判断先手胜负"""
    xor_sum = 0
    for p in piles: xor_sum ^= p
    return xor_sum != 0

def bash_game(n, m):
    """巴什博弈"""
    return n % (m + 1) != 0

def wythoff_game(a, b):
    """威佐夫博弈"""
    if a > b: a, b = b, a
    golden = (math.sqrt(5) + 1) / 2
    return a != int(golden * (b - a))

# 测试
print(nim_game([3, 4, 5]))   # True (3^4^5=2 != 0)
print(bash_game(10, 3))      # False (10 % 4 = 2 != 0, wait...)
print(wythoff_game(1, 2))    # False (奇异局势)
```

### 3.4 一般博弈状态搜索

```cpp
// 暴力搜索求博弈状态（适用于状态数较少的情况）
const int MAXN = 1005;
int state[MAXN]; // 0: 未访问, 1: N-position, -1: P-position

int solve(int x, vector<int>& moves) {
    if (state[x] != 0) return state[x];
    for (int m : moves) {
        if (x - m >= 0 && solve(x - m, moves) == -1) {
            return state[x] = 1;
        }
    }
    return state[x] = -1;
}
```

---

## 四、复杂度分析

| 博弈类型 | 判断胜负 | 求策略 |
|---------|---------|--------|
| Nim | $O(n)$ | $O(n)$ |
| 威佐夫 | $O(1)$ | - |
| 巴什 | $O(1)$ | $O(1)$ |
| 一般搜索 | $O(\text{状态数} \times \text{分支数})$ | - |

---

## 五、竞赛与面试应用场景

### 5.1 常见题型

1. **Nim 游戏变种：** 带约束的取石子
2. **阶梯 Nim：** 奇数堆做 Nim
3. **翻硬币游戏：** 对应 SG 函数
4. **图上博弈：** 在有向图上博弈

### 5.2 竞赛真题

- **洛谷 P2197：** Nim 游戏模板
- **Codeforces 博弈类：** 经常出现
- **AtCoder：** 组合博弈专题

### 5.3 注意事项

- Nim 定理是博弈论的基石，必须掌握
- 异或和为0是必败态的充要条件
- 复杂博弈需要使用 SG 函数（见下一节）
