# 概率论基础 (Probability Theory)

## 一、概念定义与原理

### 1.1 基本概念

- **样本空间 $\Omega$：** 所有可能结果的集合
- **事件 $A$：** 样本空间的子集
- **概率 $P(A)$：** 事件 $A$ 发生的可能性，$0 \leq P(A) \leq 1$
- **条件概率：** $P(A|B) = \frac{P(A \cap B)}{P(B)}$

### 1.2 基本公式

**加法公式：** $P(A \cup B) = P(A) + P(B) - P(A \cap B)$

**乘法公式：** $P(A \cap B) = P(A) \cdot P(B|A)$

**全概率公式：** $P(A) = \sum_{i} P(A|B_i) \cdot P(B_i)$

**贝叶斯公式：** $P(B_i|A) = \frac{P(A|B_i) \cdot P(B_i)}{P(A)}$

### 1.3 期望

**定义：** $E(X) = \sum_{i} x_i \cdot P(X = x_i)$

**线性性：** $E(aX + bY) = aE(X) + bE(Y)$（无论是否独立）

---

## 二、核心应用

### 2.1 期望的线性性

竞赛中最常用的性质。即使 $X, Y$ 不独立：

$$E(X + Y) = E(X) + E(Y)$$

### 2.2 方差

$$D(X) = E((X - E(X))^2) = E(X^2) - (E(X))^2$$

$D(aX + b) = a^2 D(X)$

### 2.3 独立性

$X, Y$ 独立 $\Leftrightarrow$ $P(X \cap Y) = P(X) \cdot P(Y)$ $\Leftrightarrow$ $E(XY) = E(X) \cdot E(Y)$

---

## 三、代码实现

### 3.1 期望DP - C++

```cpp
#include <bits/stdc++.h>
using namespace std;

// 经典问题：掷骰子直到和 >= n，求期望掷骰次数
// E[n] = 1 + (E[n-1] + E[n-2] + ... + E[n-6]) / 6
double expected_rolls(int n) {
    vector<double> E(n + 1, 0);
    for (int i = 1; i <= n; i++) {
        E[i] = 1.0;
        for (int j = 1; j <= 6 && i - j >= 0; j++) {
            E[i] += E[i - j] / 6.0;
        }
    }
    return E[n];
}

// 期望DP：抽奖问题
// 有 n 张卡，求集齐所有卡的期望次数
// E = n * (1/1 + 1/2 + ... + 1/n) 调和级数
double collect_all_cards(int n) {
    double result = 0;
    for (int i = 1; i <= n; i++) {
        result += (double)n / i;
    }
    return result;
}
```

### 3.2 Python 实现

```python
def expected_rolls(n):
    """掷骰子直到和 >= n 的期望次数"""
    E = [0.0] * (n + 1)
    for i in range(1, n + 1):
        E[i] = 1.0
        for j in range(1, 7):
            if i - j >= 0:
                E[i] += E[i - j] / 6.0
    return E[n]

def collect_all_cards(n):
    """集齐n张卡的期望次数（赠券收集问题）"""
    return sum(n / i for i in range(1, n + 1))

def expected_value(probs, values):
    """给定概率和对应值，计算期望"""
    return sum(p * v for p, v in zip(probs, values))

print(expected_rolls(10))        # 约 7.0
print(collect_all_cards(10))     # 约 29.29
print(expected_value([0.5, 0.3, 0.2], [1, 2, 3]))  # 1.7
```

### 3.3 概率DP模板

```cpp
// 图上随机游走求期望步数
// dp[u] = 1 + sum(dp[v] / deg[u]) 对所有邻居 v
void expected_steps(vector<vector<int>>& graph, int start, int target) {
    int n = graph.size();
    vector<double> dp(n, 0);
    // 需要用高斯消元或迭代法求解
    // 因为 dp[i] 依赖于 dp[j]，形成方程组
    // dp[i] = 1 + (1/deg[i]) * sum(dp[j])  对邻居 j
    // 整理为线性方程组后用高斯消元
}
```

---

## 四、复杂度分析

| 操作 | 时间复杂度 | 说明 |
|------|-----------|------|
| 期望DP（线性） | $O(n)$ | 递推 |
| 期望DP（图上） | $O(n^3)$ | 高斯消元 |
| 贝叶斯计算 | $O(n)$ | 遍历所有条件 |

---

## 五、竞赛与面试应用场景

### 5.1 常见题型

1. **期望DP：** 求某个随机过程的期望步数/期望得分
2. **赠券收集问题：** 集齐 $n$ 种物品的期望
3. **随机游走：** 图上从起点到终点的期望步数
4. **赌博问题：** 破产概率、期望收益

### 5.2 解题技巧

- **期望线性性是最强工具：** 将复杂随机变量拆成简单指示随机变量之和
- **期望DP的转移：** $E[S] = \sum P(\text{下一状态}) \cdot E[\text{下一状态}] + \text{当前步贡献}$
- **无穷级数：** 几何分布期望 $E = 1/p$

### 5.3 注意事项

- 期望线性性不需要独立性
- 方差不满足线性性（除非独立）
- 浮点精度注意误差积累
