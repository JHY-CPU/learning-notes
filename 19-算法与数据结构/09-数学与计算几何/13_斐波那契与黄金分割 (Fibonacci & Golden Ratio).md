# 斐波那契与黄金分割 (Fibonacci & Golden Ratio)

## 一、概念定义与原理

### 1.1 斐波那契数列

**定义：** $F_0 = 0, F_1 = 1, F_n = F_{n-1} + F_{n-2}$（$n \geq 2$）

数列：$0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, \ldots$

### 1.2 黄金分割比

$$\varphi = \frac{1 + \sqrt{5}}{2} \approx 1.6180339887\ldots$$

其共轭：$\hat{\varphi} = \frac{1 - \sqrt{5}}{2} \approx -0.618\ldots$

### 1.3 通项公式（Binet 公式）

$$F_n = \frac{\varphi^n - \hat{\varphi}^n}{\sqrt{5}} = \frac{1}{\sqrt{5}}\left[\left(\frac{1+\sqrt{5}}{2}\right)^n - \left(\frac{1-\sqrt{5}}{2}\right)^n\right]$$

由于 $|\hat{\varphi}| < 1$，当 $n$ 较大时 $F_n \approx \frac{\varphi^n}{\sqrt{5}}$。

---

## 二、核心性质

### 2.1 基本性质

1. **通项：** $F_n = \frac{\varphi^n - \hat{\varphi}^n}{\sqrt{5}}$
2. **Cassini恒等式：** $F_{n-1} \cdot F_{n+1} - F_n^2 = (-1)^n$
3. **和公式：** $\sum_{i=0}^{n} F_i = F_{n+2} - 1$
4. **平方和：** $\sum_{i=0}^{n} F_i^2 = F_n \cdot F_{n+1}$
5. **GCD性质：** $\gcd(F_m, F_n) = F_{\gcd(m, n)}$
6. **整除性：** $F_m \mid F_n$ 当且仅当 $m \mid n$

### 2.2 矩阵表示

$$\begin{pmatrix} F_{n+1} \\ F_n \end{pmatrix} = \begin{pmatrix} 1 & 1 \\ 1 & 0 \end{pmatrix}^n \begin{pmatrix} 1 \\ 0 \end{pmatrix}$$

利用矩阵快速幂可在 $O(\log n)$ 时间内求 $F_n$。

### 2.3 循环节

$F_n \bmod m$ 是周期的。对于质数 $p \neq 5$，周期整除 $p-1$ 或 $2(p+1)$。

---

## 三、代码实现

### 3.1 直接计算 - C++

```cpp
#include <bits/stdc++.h>
using namespace std;

// O(n) 递推
long long fibonacci_iter(int n) {
    if (n <= 1) return n;
    long long a = 0, b = 1;
    for (int i = 2; i <= n; i++) {
        long long c = a + b;
        a = b; b = c;
    }
    return b;
}
```

### 3.2 矩阵快速幂 $O(\log n)$ - C++

```cpp
struct Matrix {
    long long a[2][2];
    Matrix() { memset(a, 0, sizeof(a)); }
    Matrix operator*(const Matrix& B) const {
        Matrix C;
        for (int i = 0; i < 2; i++)
            for (int k = 0; k < 2; k++)
                for (int j = 0; j < 2; j++)
                    C.a[i][j] += a[i][k] * B.a[k][j];
        return C;
    }
};

long long fibonacci_matrix(long long n, long long mod) {
    if (n <= 1) return n;
    Matrix A, R;
    A.a[0][0] = A.a[0][1] = A.a[1][0] = 1;
    A.a[1][1] = 0;
    R.a[0][0] = R.a[1][1] = 1;
    long long p = n - 1;
    while (p) {
        if (p & 1) R = R * A;
        A = A * A;
        p >>= 1;
    }
    return R.a[0][0] % mod;
}
```

### 3.3 Python 实现

```python
import math

def fibonacci_binet(n):
    """Binet公式（有精度误差，仅适用于小n）"""
    phi = (1 + math.sqrt(5)) / 2
    psi = (1 - math.sqrt(5)) / 2
    return round((phi**n - psi**n) / math.sqrt(5))

def fibonacci_matrix(n, mod):
    """矩阵快速幂 O(log n)"""
    if n <= 1: return n
    def mat_mul(A, B):
        return [[(A[0][0]*B[0][0]+A[0][1]*B[1][0])%mod,
                 (A[0][0]*B[0][1]+A[0][1]*B[1][1])%mod],
                [(A[1][0]*B[0][0]+A[1][1]*B[1][0])%mod,
                 (A[1][0]*B[0][1]+A[1][1]*B[1][1])%mod]]
    A = [[1, 1], [1, 0]]
    R = [[1, 0], [0, 1]]
    p = n - 1
    while p:
        if p & 1: R = mat_mul(R, A)
        A = mat_mul(A, A)
        p >>= 1
    return R[0][0]

MOD = 10**9 + 7
print(fibonacci_matrix(10, MOD))       # 55
print(fibonacci_matrix(10**18, MOD))    # 快速计算
```

### 3.4 求循环节

```cpp
// 求 Fibonacci 数列模 m 的 Pisano 周期
long long pisano_period(long long m) {
    long long a = 0, b = 1;
    for (long long i = 1; i <= 6 * m; i++) {
        long long c = (a + b) % m;
        a = b; b = c;
        if (a == 0 && b == 1) return i;
    }
    return -1;
}
```

---

## 四、复杂度分析

| 方法 | 时间复杂度 | 空间复杂度 | 适用场景 |
|------|-----------|-----------|---------|
| 递推 | $O(n)$ | $O(1)$ | $n \leq 10^7$ |
| Binet公式 | $O(1)$ | $O(1)$ | 有精度误差 |
| 矩阵快速幂 | $O(\log n)$ | $O(1)$ | 通用 |
| 通项取模 | 需处理 $\sqrt{5}$ | - | 竞赛中少用 |

---

## 五、竞赛与面试应用场景

1. **斐波那契第n项取模：** 矩阵快速幂
2. **斐波那契性质：** GCD、整除性质
3. **螺旋结构：** 自然界中的斐波那契现象
4. **卡特兰数关联：** $C_n = \frac{1}{n+1}\binom{2n}{n}$
