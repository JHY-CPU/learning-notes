# 数学算法总结 (Math Algorithms Summary)

## 一、知识体系总览

本节总结竞赛中常用数学算法的知识体系，提供选型指南。

### 1.1 数论模块

| 算法 | 应用 | 复杂度 | 关键词 |
|------|------|--------|--------|
| 质数判定 | 判断素性 | $O(\sqrt{n})$ / Miller-Rabin $O(\log^2 n)$ | 试除法/Miller-Rabin |
| 质因数分解 | 因子分析 | $O(\sqrt{n})$ / Pollard Rho $O(n^{1/4})$ | 试除法/Pollard |
| GCD/LCM | 最大公约数 | $O(\log n)$ | 欧几里得 |
| 快速幂 | $a^b \bmod m$ | $O(\log b)$ | 二进制拆分 |
| 筛法 | 批量素数 | $O(n)$ | 欧拉筛 |
| 扩展欧几里得 | $ax+by=gcd$ | $O(\log n)$ | 模逆元/CRT |
| 欧拉函数 | $\varphi(n)$ | $O(\sqrt{n})$ / $O(n)$ 线性筛 | 积性函数 |
| CRT/EXCRT | 同余方程组 | $O(k \log m)$ | 中国剩余 |

### 1.2 组合数学模块

| 算法 | 应用 | 复杂度 | 关键词 |
|------|------|--------|--------|
| 组合数取模 | $C_n^m \bmod p$ | $O(n)$ 预处理 | 阶乘+逆元 |
| Lucas 定理 | 大组合数模小质数 | $O(p \log_p n)$ | 分治 |
| 容斥原理 | 并集计数 | $O(2^k)$ | 子集枚举 |
| 莫比乌斯反演 | 互质计数等 | $O(n)$ / $O(\sqrt{n})$ 分块 | $\mu$ 函数 |
| 卡特兰数 | 括号/路径 | $O(n)$ 预处理 | $C_n = \frac{1}{n+1}\binom{2n}{n}$ |

### 1.3 线性代数模块

| 算法 | 应用 | 复杂度 | 关键词 |
|------|------|--------|--------|
| 矩阵快速幂 | 线性递推 | $O(k^3 \log n)$ | 转移矩阵 |
| 高斯消元 | 线性方程组 | $O(n^3)$ | 行阶梯形 |
| 行列式计算 | 多种应用 | $O(n^3)$ | 消元过程 |

### 1.4 博弈论模块

| 算法 | 应用 | 复杂度 | 关键词 |
|------|------|--------|--------|
| Nim 定理 | 取石子 | $O(n)$ | 异或和 |
| SG 函数 | 复杂博弈 | 状态相关 | mex 运算 |

---

## 二、选型指南

### 2.1 按问题类型选择

**"求 $a^b \bmod m$"：**
- $m$ 为质数：费马小定理 + 快速幂
- $m$ 非质数：扩展欧几里得求逆元
- $b$ 极大：降幂公式

**"组合数取模"：**
- $n \leq 10^6$，$p$ 为质数：预处理阶乘+逆元
- $n$ 极大，$p$ 小质数：Lucas 定理
- $p$ 非质数：质因数分解法

**"互质对计数"：**
- 容斥原理（$k \leq 20$）
- 莫比乌斯反演（一般情况）

**"线性递推"：**
- 矩阵快速幂 $O(k^3 \log n)$

**"博弈判断"：**
- 简单取石子：Nim 定理（异或）
- 复杂博弈：SG 函数

### 2.2 常见陷阱

1. **溢出：** 乘法中间结果可能溢出 `long long`，需用 `__int128` 或先除后乘
2. **精度：** 浮点运算注意误差，用 `eps` 比较
3. **取模：** 注意负数取模需要加模数
4. **特判：** $\gcd(0, n) = n$，$0! = 1$ 等边界

---

## 三、代码速查

### 3.1 模板速查

```cpp
// 快速幂
long long qpow(long long a, long long b, long long m) {
    long long r = 1; a %= m;
    while (b) { if (b&1) r=r*a%m; a=a*a%m; b>>=1; }
    return r;
}

// GCD
long long gcd(long long a, long long b) { return b ? gcd(b, a%b) : a; }

// 扩展欧几里得
long long exgcd(long long a, long long b, long long &x, long long &y) {
    if (!b) { x=1; y=0; return a; }
    long long g = exgcd(b, a%b, y, x); y -= a/b*x; return g;
}

// 组合数（预处理后）
const int MOD = 1e9+7, N = 2e6+5;
long long fac[N], inv_fac[N];
void init() { fac[0]=1; for(int i=1;i<N;i++) fac[i]=fac[i-1]*i%MOD;
    inv_fac[N-1]=qpow(fac[N-1],MOD-2,MOD);
    for(int i=N-2;i>=0;i--) inv_fac[i]=inv_fac[i+1]*(i+1)%MOD; }
long long C(int n, int m) { return fac[n]*inv_fac[m]%MOD*inv_fac[n-m]%MOD; }
```

---

## 四、竞赛中的数学技巧

### 4.1 加速技巧

- **分块处理：** 对于 $\lfloor n/d \rfloor$ 的枚举，$d$ 相同时值相同，可分 $O(\sqrt{n})$ 段
- **记忆化：** 递归计算的中间结果缓存
- **预处理：** 质数表、阶乘表、欧拉函数表等

### 4.2 常见递推公式

- **Catalan：** $C_n = \frac{2(2n-1)}{n+1} C_{n-1}$
- **Stirling 第二类：** $S(n,k) = k \cdot S(n-1,k) + S(n-1,k-1)$
- **Bell 数：** $B_n = \sum_{k=0}^{n-1} \binom{n-1}{k} B_k$
