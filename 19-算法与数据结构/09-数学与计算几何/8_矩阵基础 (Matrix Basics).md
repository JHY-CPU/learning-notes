# 矩阵基础 (Matrix Basics)

## 一、概念定义与原理

### 1.1 矩阵定义

矩阵是一个 $m \times n$ 的数表，记为 $A_{m \times n}$ 或 $A = (a_{ij})_{m \times n}$。

### 1.2 矩阵运算

**加法：** 同型矩阵对应元素相加，$O(mn)$

**乘法：** $C_{m \times p} = A_{m \times n} \cdot B_{n \times p}$，$c_{ij} = \sum_{k=1}^{n} a_{ik} \cdot b_{kj}$，时间复杂度 $O(mnp)$

**转置：** $A^T$ 的第 $(i,j)$ 元素为 $A$ 的第 $(j,i)$ 元素

### 1.3 特殊矩阵

- **单位矩阵 $I$：** 主对角线为1，其余为0，$AI = IA = A$
- **零矩阵 $O$：** 所有元素为0
- **对角矩阵：** 非对角元素全为0

---

## 二、核心性质

### 2.1 矩阵乘法性质

- **结合律：** $(AB)C = A(BC)$
- **分配律：** $A(B+C) = AB + AC$
- **不满足交换律：** $AB \neq BA$（一般情况）

### 2.2 矩阵行列式

$2 \times 2$：$\det\begin{pmatrix} a & b \\ c & d \end{pmatrix} = ad - bc$

### 2.3 逆矩阵

若 $AB = BA = I$，则 $B = A^{-1}$。求逆用增广矩阵 $(A | I)$ 做行变换得到 $(I | A^{-1})$。

---

## 三、代码实现

### 3.1 矩阵结构体 - C++

```cpp
#include <bits/stdc++.h>
using namespace std;

struct Matrix {
    vector<vector<long long>> a;
    int rows, cols;

    Matrix(int r, int c) : rows(r), cols(c) {
        a.assign(r, vector<long long>(c, 0));
    }

    static Matrix identity(int n) {
        Matrix I(n, n);
        for (int i = 0; i < n; i++) I.a[i][i] = 1;
        return I;
    }

    Matrix operator*(const Matrix& other) const {
        Matrix result(rows, other.cols);
        for (int i = 0; i < rows; i++)
            for (int k = 0; k < cols; k++)
                if (a[i][k] != 0)
                    for (int j = 0; j < other.cols; j++)
                        result.a[i][j] += a[i][k] * other.a[k][j];
        return result;
    }

    Matrix mul(const Matrix& other, long long mod) const {
        Matrix result(rows, other.cols);
        for (int i = 0; i < rows; i++)
            for (int k = 0; k < cols; k++)
                for (int j = 0; j < other.cols; j++)
                    result.a[i][j] = (result.a[i][j] + a[i][k] * other.a[k][j]) % mod;
        return result;
    }
};
```

### 3.2 矩阵快速幂

```cpp
Matrix matrix_power(Matrix A, long long p, long long mod) {
    Matrix result = Matrix::identity(A.rows);
    while (p > 0) {
        if (p & 1) result = result.mul(A, mod);
        A = A.mul(A, mod);
        p >>= 1;
    }
    return result;
}
```

### 3.3 Python 实现

```python
class Matrix:
    def __init__(self, data):
        self.data = [row[:] for row in data]
        self.rows = len(data)
        self.cols = len(data[0])

    @staticmethod
    def identity(n):
        data = [[0] * n for _ in range(n)]
        for i in range(n): data[i][i] = 1
        return Matrix(data)

    def mul_mod(self, other, mod):
        result = [[0] * other.cols for _ in range(self.rows)]
        for i in range(self.rows):
            for k in range(self.cols):
                for j in range(other.cols):
                    result[i][j] = (result[i][j] + self.data[i][k] * other.data[k][j]) % mod
        return Matrix(result)

    def power(self, p, mod):
        result = Matrix.identity(self.rows)
        base = Matrix([row[:] for row in self.data])
        while p > 0:
            if p & 1: result = result.mul_mod(base, mod)
            base = base.mul_mod(base, mod)
            p >>= 1
        return result

# 测试：斐波那契 F(10)
# [F(n+1), F(n)]^T = [[1,1],[1,0]]^n * [1,0]^T
M = Matrix([[1, 1], [1, 0]])
print(M.power(10, 10**9+7).data[0][1])  # 55
```

---

## 四、复杂度分析

| 操作 | 时间复杂度 | 空间复杂度 |
|------|-----------|-----------|
| 矩阵乘法 | $O(n^3)$ 朴素 | $O(n^2)$ |
| Strassen | $O(n^{2.807})$ | $O(n^2)$ |
| 矩阵快速幂 | $O(n^3 \log p)$ | $O(n^2)$ |

---

## 五、竞赛与面试应用场景

1. **矩阵快速幂：** 加速线性递推，如斐波那契 $O(\log n)$
2. **图的邻接矩阵：** $A^k$ 的 $(i,j)$ 元表示从 $i$ 到 $j$ 长度为 $k$ 的路径数
3. **高斯消元：** 解线性方程组
4. **坐标变换：** 旋转、缩放、平移

**优化技巧：** 循环顺序 $i \to k \to j$ 比 $i \to j \to k$ 更缓存友好。
