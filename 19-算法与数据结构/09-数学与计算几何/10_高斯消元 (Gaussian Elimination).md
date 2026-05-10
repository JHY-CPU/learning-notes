# 高斯消元 (Gaussian Elimination)

## 一、概念定义与原理

### 1.1 核心思想

高斯消元是求解线性方程组的经典算法，通过**初等行变换**将增广矩阵化为**行阶梯形**或**行最简形**，然后回代求解。

### 1.2 初等行变换

三种操作不改变方程组的解：
1. 交换两行
2. 某行乘以非零常数
3. 某行加上另一行的倍数

### 1.3 行阶梯形条件

- 零行在底部
- 每行首非零元素（主元）所在列，上方行的主元在其左边

---

## 二、核心算法

### 2.1 标准高斯消元

1. **选主元：** 在当前列找绝对值最大的元素作为主元（部分选主元法，提高数值稳定性）
2. **交换行：** 将主元所在行交换到当前行
3. **消元：** 将当前列下方所有元素消为0
4. **回代：** 从最后一行向上求解

### 2.2 特殊情况

- **无解：** 出现 $0 = c$（$c \neq 0$）的矛盾方程
- **无穷多解：** 有自由变量（主元数 < 未知数数）

---

## 三、代码实现

### 3.1 实数域高斯消元 - C++

```cpp
#include <bits/stdc++.h>
using namespace std;

// 求解 Ax = b，返回 1 唯一解, 0 无穷多解, -1 无解
int gauss(vector<vector<double>>& A, vector<double>& b, vector<double>& x) {
    int n = A.size();
    // 构造增广矩阵
    for (int i = 0; i < n; i++) A[i].push_back(b[i]);

    for (int col = 0; col < n; col++) {
        // 部分选主元
        int pivot = col;
        for (int row = col + 1; row < n; row++) {
            if (abs(A[row][col]) > abs(A[pivot][col])) pivot = row;
        }
        swap(A[col], A[pivot]);
        if (abs(A[col][col]) < 1e-9) continue;

        // 消元
        for (int row = col + 1; row < n; row++) {
            double factor = A[row][col] / A[col][col];
            for (int j = col; j <= n; j++) {
                A[row][j] -= factor * A[col][j];
            }
        }
    }

    // 回代
    x.assign(n, 0);
    for (int i = n - 1; i >= 0; i--) {
        if (abs(A[i][i]) < 1e-9) {
            if (abs(A[i][n]) > 1e-9) return -1;
            return 0;
        }
        x[i] = A[i][n];
        for (int j = i + 1; j < n; j++) x[i] -= A[i][j] * x[j];
        x[i] /= A[i][i];
    }
    return 1;
}
```

### 3.2 模意义下高斯消元 - C++

```cpp
// 在模 p 意义下求解 (p 为质数)
int gauss_mod(vector<vector<long long>>& A, vector<long long>& b,
              vector<long long>& x, long long p) {
    int n = A.size();
    for (int i = 0; i < n; i++) A[i].push_back(b[i]);

    for (int col = 0; col < n; col++) {
        int pivot = -1;
        for (int row = col; row < n; row++) {
            if (A[row][col] % p != 0) { pivot = row; break; }
        }
        if (pivot == -1) continue;
        swap(A[col], A[pivot]);

        long long inv = power(A[col][col], p - 2, p);
        for (int row = col + 1; row < n; row++) {
            long long factor = A[row][col] % p * inv % p;
            for (int j = col; j <= n; j++) {
                A[row][j] = ((A[row][j] - factor * A[col][j]) % p + p) % p;
            }
        }
    }

    x.assign(n, 0);
    for (int i = n - 1; i >= 0; i--) {
        if (A[i][i] % p == 0) {
            if (A[i][n] % p != 0) return -1;
            return 0;
        }
        x[i] = A[i][n] % p * power(A[i][i], p - 2, p) % p;
        for (int j = i + 1; j < n; j++) {
            x[i] = ((x[i] - A[i][j] % p * x[j]) % p + p) % p;
        }
    }
    return 1;
}
```

### 3.3 Python 实现

```python
def gauss(A, b):
    """求解 Ax = b"""
    n = len(A)
    aug = [A[i][:] + [b[i]] for i in range(n)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
        aug[col], aug[pivot] = aug[pivot], aug[col]
        if abs(aug[col][col]) < 1e-12: continue
        for row in range(col + 1, n):
            factor = aug[row][col] / aug[col][col]
            for j in range(col, n + 1):
                aug[row][j] -= factor * aug[col][j]
    x = [0] * n
    for i in range(n - 1, -1, -1):
        x[i] = aug[i][n]
        for j in range(i + 1, n): x[i] -= aug[i][j] * x[j]
        x[i] /= aug[i][i]
    return x

# 测试
A = [[2, 1], [1, 3]]
b = [5, 10]
print(gauss(A, b))  # [1.0, 3.0]
```

---

## 四、复杂度分析

| 操作 | 时间复杂度 | 空间复杂度 |
|------|-----------|-----------|
| 消元过程 | $O(n^3)$ | $O(n^2)$ |
| 回代 | $O(n^2)$ | $O(n)$ |
| 部分选主元 | 额外 $O(n^2)$ | $O(1)$ |

---

## 五、竞赛与面试应用场景

1. **线性方程组求解：** 物理/工程问题建模
2. **行列式计算：** 消元过程中主元乘积
3. **矩阵求逆：** 对 $(A|I)$ 做行变换
4. **博弈论期望：** 高斯消元求期望值
5. **异或方程组：** 模2的高斯消元
