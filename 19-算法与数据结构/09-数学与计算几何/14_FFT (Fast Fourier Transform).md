# 快速傅里叶变换 (Fast Fourier Transform, FFT)

## 一、概念定义与原理

### 1.1 问题背景

**多项式乘法：** 给定两个多项式 $A(x)$ 和 $B(x)$，求 $C(x) = A(x) \cdot B(x)$。

- 系数表示法下的朴素乘法：$O(n^2)$
- 点值表示法下的乘法：$O(n)$（若已知 $n+1$ 个点值）
- FFT 将系数表示转为点值表示，相乘后再转回：$O(n \log n)$

### 1.2 核心思想

**DFT（离散傅里叶变换）：** 将系数表示转换为点值表示

**IDFT（逆变换）：** 将点值表示转换为系数表示

利用单位根的特殊性质，FFT 通过分治在 $O(n \log n)$ 时间内完成变换。

### 1.3 单位根

$n$ 次单位根 $\omega_n = e^{2\pi i / n} = \cos\frac{2\pi}{n} + i\sin\frac{2\pi}{n}$

**性质：**
- $\omega_n^k = \omega_n^{k \bmod n}$
- $\omega_n^{k + n/2} = -\omega_n^k$
- $\omega_{2n}^{2k} = \omega_n^k$

---

## 二、核心算法

### 2.1 Cooley-Tukey FFT

将 $n$ 次多项式按奇偶拆分：

$$A(x) = A_0(x^2) + x \cdot A_1(x^2)$$

其中 $A_0$ 和 $A_1$ 分别为偶次项和奇次项组成的多项式。

代入 $\omega_n^k$：

$$A(\omega_n^k) = A_0(\omega_{n/2}^k) + \omega_n^k \cdot A_1(\omega_{n/2}^k)$$

$$A(\omega_n^{k+n/2}) = A_0(\omega_{n/2}^k) - \omega_n^k \cdot A_1(\omega_{n/2}^k)$$

### 2.2 逆变换

IDFT 只需将单位根取共轭（即 $\omega_n^{-1}$），结果除以 $n$。

---

## 三、代码实现

### 3.1 C++ 实现（复数版）

```cpp
#include <bits/stdc++.h>
using namespace std;

const double PI = acos(-1.0);

struct Complex {
    double r, i;
    Complex(double r = 0, double i = 0) : r(r), i(i) {}
    Complex operator+(const Complex& b) const { return {r + b.r, i + b.i}; }
    Complex operator-(const Complex& b) const { return {r - b.r, i - b.i}; }
    Complex operator*(const Complex& b) const { return {r*b.r - i*b.i, r*b.i + i*b.r}; }
};

void fft(vector<Complex>& a, int inv) {
    int n = a.size();
    // 位逆序置换
    for (int i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) swap(a[i], a[j]);
    }
    // 分治
    for (int len = 2; len <= n; len <<= 1) {
        double ang = 2 * PI / len * inv;
        Complex wlen(cos(ang), sin(ang));
        for (int i = 0; i < n; i += len) {
            Complex w(1, 0);
            for (int j = 0; j < len / 2; j++) {
                Complex u = a[i + j], v = a[i + j + len/2] * w;
                a[i + j] = u + v;
                a[i + j + len/2] = u - v;
                w = w * wlen;
            }
        }
    }
    if (inv == -1) for (auto& x : a) x.r /= n;
}

// 多项式乘法
vector<long long> multiply(vector<int>& a, vector<int>& b) {
    vector<Complex> fa(a.begin(), a.end()), fb(b.begin(), b.end());
    int n = 1;
    while (n < a.size() + b.size()) n <<= 1;
    fa.resize(n); fb.resize(n);
    fft(fa, 1); fft(fb, 1);
    for (int i = 0; i < n; i++) fa[i] = fa[i] * fb[i];
    fft(fa, -1);
    vector<long long> result(n);
    for (int i = 0; i < n; i++) result[i] = round(fa[i].r);
    return result;
}
```

### 3.2 Python 实现

```python
import cmath

def fft(a, inv=1):
    n = len(a)
    j = 0
    for i in range(1, n):
        bit = n >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j ^= bit
        if i < j: a[i], a[j] = a[j], a[i]
    length = 2
    while length <= n:
        wlen = cmath.exp(2j * cmath.pi / length * inv)
        for i in range(0, n, length):
            w = 1
            for j in range(length // 2):
                u = a[i + j]
                v = a[i + j + length // 2] * w
                a[i + j] = u + v
                a[i + j + length // 2] = u - v
                w *= wlen
        length <<= 1
    if inv == -1:
        for i in range(n): a[i] /= n

def multiply(a, b):
    n = 1
    while n < len(a) + len(b): n <<= 1
    fa = list(map(complex, a)) + [0] * (n - len(a))
    fb = list(complex, b) + [0] * (n - len(b))
    fft(fa, 1); fft(fb, 1)
    for i in range(n): fa[i] *= fb[i]
    fft(fa, -1)
    return [round(fa[i].real) for i in range(n)]

print(multiply([1, 2, 3], [4, 5]))  # [4, 13, 22, 15]
```

---

## 四、复杂度分析

| 操作 | 时间复杂度 | 空间复杂度 |
|------|-----------|-----------|
| FFT | $O(n \log n)$ | $O(n)$ |
| IFFT | $O(n \log n)$ | $O(n)$ |
| 多项式乘法 | $O(n \log n)$ | $O(n)$ |

---

## 五、竞赛与面试应用场景

1. **大整数乘法：** 将数字拆成系数做多项式乘法
2. **卷积运算：** 信号处理、模式匹配
3. **字符串匹配：** 利用FFT加速
4. **组合计数：** 某些递推的加速
5. **NTT（数论变换）：** 在模意义下的FFT，避免浮点误差
