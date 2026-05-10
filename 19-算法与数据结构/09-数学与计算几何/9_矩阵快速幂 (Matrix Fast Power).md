# 矩阵快速幂 (Matrix Fast Power)

## 一、概念定义与原理

### 1.1 核心思想

将快速幂的思想推广到矩阵运算。对于矩阵 $A$ 和正整数 $n$，利用二进制拆分在 $O(k^3 \log n)$ 时间内计算 $A^n$（$k$ 为矩阵阶数）。

### 1.2 经典应用：斐波那契数列

斐波那契递推 $F(n) = F(n-1) + F(n-2)$ 可以写成矩阵形式：

$$\begin{pmatrix} F(n+1) \\ F(n) \end{pmatrix} = \begin{pmatrix} 1 & 1 \\ 1 & 0 \end{pmatrix}^n \begin{pmatrix} 1 \\ 0 \end{pmatrix}$$

因此 $F(n)$ 可以在 $O(\log n)$ 时间内求出。

### 1.3 适用条件

任何**线性递推关系**都可以用矩阵快速幂加速。关键在于构造正确的转移矩阵。

---

## 二、核心算法

### 2.1 矩阵快速幂

与标量快速幂类似：
- $A^n = A^{n/2} \cdot A^{n/2}$（$n$ 为偶数）
- $A^n = A^{n/2} \cdot A^{n/2} \cdot A$（$n$ 为奇数）

### 2.2 常见递推的矩阵构造

**斐波那契：**
$$\begin{pmatrix} F_{n+1} \\ F_n \end{pmatrix} = \begin{pmatrix} 1 & 1 \\ 1 & 0 \end{pmatrix}^n \begin{pmatrix} 1 \\ 0 \end{pmatrix}$$

**k阶线性递推：** $a_n = c_1 a_{n-1} + c_2 a_{n-2} + \cdots + c_k a_{n-k}$

$$\begin{pmatrix} a_n \\ a_{n-1} \\ \vdots \\ a_{n-k+1} \end{pmatrix} = \begin{pmatrix} c_1 & c_2 & \cdots & c_k \\ 1 & 0 & \cdots & 0 \\ 0 & 1 & \cdots & 0 \\ \vdots & & \ddots & \\ 0 & 0 & \cdots & 1 \end{pmatrix} \begin{pmatrix} a_{n-1} \\ a_{n-2} \\ \vdots \\ a_{n-k} \end{pmatrix}$$

---

## 三、代码实现

### 3.1 C++ 完整实现

```cpp
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

struct Matrix {
    vector<vector<ll>> a;
    int n;
    Matrix(int sz) : n(sz) { a.assign(sz, vector<ll>(sz, 0)); }
    static Matrix identity(int sz) {
        Matrix I(sz);
        for (int i = 0; i < sz; i++) I.a[i][i] = 1;
        return I;
    }
    Matrix mul(const Matrix& B, ll mod) const {
        Matrix C(n);
        for (int i = 0; i < n; i++)
            for (int k = 0; k < n; k++)
                for (int j = 0; j < n; j++)
                    C.a[i][j] = (C.a[i][j] + a[i][k] * B.a[k][j]) % mod;
        return C;
    }
};

Matrix mat_pow(Matrix A, ll p, ll mod) {
    Matrix result = Matrix::identity(A.n);
    while (p) {
        if (p & 1) result = result.mul(A, mod);
        A = A.mul(A, mod);
        p >>= 1;
    }
    return result;
}

// 求 F(n) mod mod
ll fibonacci(ll n, ll mod) {
    if (n <= 1) return n;
    Matrix A(2);
    A.a[0][0] = 1; A.a[0][1] = 1;
    A.a[1][0] = 1; A.a[1][1] = 0;
    Matrix An = mat_pow(A, n - 1, mod);
    return An.a[0][0]; // F(n)
}
```

### 3.2 Python 实现

```python
def mat_mul(A, B, mod):
    n = len(A)
    C = [[0] * n for _ in range(n)]
    for i in range(n):
        for k in range(n):
            for j in range(n):
                C[i][j] = (C[i][j] + A[i][k] * B[k][j]) % mod
    return C

def mat_pow(A, p, mod):
    n = len(A)
    result = [[int(i == j) for j in range(n)] for i in range(n)]
    while p:
        if p & 1:
            result = mat_mul(result, A, mod)
        A = mat_mul(A, A, mod)
        p >>= 1
    return result

def fibonacci(n, mod):
    if n <= 1: return n
    A = [[1, 1], [1, 0]]
    An = mat_pow(A, n - 1, mod)
    return An[0][0]

MOD = 10**9 + 7
print(fibonacci(10, MOD))      # 55
print(fibonacci(1000000000, MOD))  # 快速计算大数
```

### 3.3 k阶递推模板

```cpp
// a_n = c1*a_{n-1} + c2*a_{n-2} + ... + ck*a_{n-k}
// 给定 a[0..k-1]，求 a[n]
ll linear_recurrence(vector<ll>& init, vector<ll>& coeff, ll n, ll mod) {
    int k = init.size();
    if (n < k) return init[n];
    Matrix A(k);
    for (int i = 0; i < k; i++) A.a[0][i] = coeff[i];
    for (int i = 1; i < k; i++) A.a[i][i-1] = 1;
    Matrix An = mat_pow(A, n - k + 1, mod);
    ll result = 0;
    for (int i = 0; i < k; i++) {
        result = (result + An.a[0][i] * init[k - 1 - i]) % mod;
    }
    return result;
}
```

---

## 四、复杂度分析

| 操作 | 时间复杂度 | 说明 |
|------|-----------|------|
| 矩阵乘法 | $O(k^3)$ | $k$ 为矩阵阶数 |
| 矩阵快速幂 | $O(k^3 \log n)$ | $\log n$ 次矩阵乘法 |
| 斐波那契 | $O(\log n)$ | $k=2$，常数很小 |

---

## 五、竞赛与面试应用场景

### 5.1 常见题型

1. **斐波那契第n项：** $O(\log n)$ 求 $F(n) \bmod p$
2. **线性递推加速：** 任何k阶线性递推
3. **路径计数：** 邻接矩阵的n次幂
4. **带约束计数：** 状态转移用矩阵表示

### 5.2 竞赛真题

- **洛谷 P3390：** 矩阵快速幂模板
- **洛谷 P1939：** 矩阵加速递推（3阶递推）
- **Codeforces：** 矩阵快速幂 + 组合数学

### 5.3 注意事项

- 矩阵阶数 $k$ 通常很小（$\leq 100$），$O(k^3 \log n)$ 可接受
- 取模时注意使用 `long long` 防止中间溢出
- 指数 $n$ 可以很大（如 $10^{18}$），$\log n$ 仅约 60 次
