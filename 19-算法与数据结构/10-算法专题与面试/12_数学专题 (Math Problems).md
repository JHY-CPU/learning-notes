# 数学专题 (Math Problems)

## 一、数论基础

### 1.1 素数

**埃拉托斯特尼筛法：** $O(n \log \log n)$ 时间生成素数表。

```python
def sieve(n):
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, n+1, i):
                is_prime[j] = False
    return [i for i in range(n+1) if is_prime[i]]
```

**素性检测：** 试除法 $O(\sqrt{n})$

```python
def is_prime(n):
    if n < 2: return False
    if n < 4: return True
    if n % 2 == 0 or n % 3 == 0: return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i+2) == 0:
            return False
        i += 6
    return True
```

### 1.2 最大公约数与最小公倍数

```python
def gcd(a, b):
    while b: a, b = b, a % b
    return a

def lcm(a, b):
    return a * b // gcd(a, b)
```

### 1.3 快速幂

```python
def fast_pow(base, exp, mod=10**9+7):
    result = 1
    base %= mod
    while exp > 0:
        if exp & 1:
            result = result * base % mod
        base = base * base % mod
        exp >>= 1
    return result
```

**C++ 实现：**

```cpp
long long fastPow(long long base, long long exp, long long mod) {
    long long result = 1;
    base %= mod;
    while (exp > 0) {
        if (exp & 1) result = result * base % mod;
        base = base * base % mod;
        exp >>= 1;
    }
    return result;
}
```

---

## 二、组合数学

### 2.1 组合数计算

```python
def comb(n, k):
    if k > n: return 0
    k = min(k, n - k)
    result = 1
    for i in range(k):
        result = result * (n - i) // (i + 1)
    return result
```

**预处理阶乘（多次查询）：**

```python
MOD = 10**9 + 7
MAXN = 10**5 + 1

fact = [1] * MAXN
inv_fact = [1] * MAXN

for i in range(1, MAXN):
    fact[i] = fact[i-1] * i % MOD

inv_fact[MAXN-1] = pow(fact[MAXN-1], MOD-2, MOD)
for i in range(MAXN-2, -1, -1):
    inv_fact[i] = inv_fact[i+1] * (i+1) % MOD

def comb_precomputed(n, k):
    if k < 0 or k > n: return 0
    return fact[n] * inv_fact[k] % MOD * inv_fact[n-k] % MOD
```

### 2.2 卡特兰数

$$C_n = \frac{1}{n+1}\binom{2n}{n}$$

应用：合法括号数、二叉树个数、凸多边形三角剖分。

```python
def catalan(n):
    return comb(2*n, n) // (n + 1)
```

---

## 三、进制转换

### 3.1 进制转换实现

```python
def to_base(n, base):
    if n == 0: return "0"
    digits = "0123456789ABCDEF"
    result = []
    while n > 0:
        result.append(digits[n % base])
        n //= base
    return ''.join(reversed(result))

def from_base(s, base):
    result = 0
    for c in s:
        result = result * base + int(c, base)
    return result
```

---

## 四、其他数学技巧

### 4.1 丑数 (LeetCode 263/264)

```python
def nth_ugly_number(n):
    ugly = [1]
    i2 = i3 = i5 = 0
    for _ in range(1, n):
        next_ugly = min(ugly[i2]*2, ugly[i3]*3, ugly[i5]*5)
        ugly.append(next_ugly)
        if next_ugly == ugly[i2]*2: i2 += 1
        if next_ugly == ugly[i3]*3: i3 += 1
        if next_ugly == ugly[i5]*5: i5 += 1
    return ugly[-1]
```

### 4.2 阶乘后的零 (LeetCode 172)

```python
def trailing_zeros(n):
    count = 0
    while n >= 5:
        n //= 5
        count += n
    return count
```

### 4.3 整数反转 (LeetCode 7)

```python
def reverse_integer(x):
    result = 0
    sign = 1 if x >= 0 else -1
    x = abs(x)
    while x:
        result = result * 10 + x % 10
        x //= 10
    result *= sign
    if result > 2**31 - 1 or result < -2**31:
        return 0
    return result
```

### 4.4 罗马数字 (LeetCode 12/13)

```python
def int_to_roman(num):
    values = [1000,900,500,400,100,90,50,40,10,9,5,4,1]
    symbols = ["M","CM","D","CD","C","XC","L","XL","X","IX","V","IV","I"]
    result = []
    for v, s in zip(values, symbols):
        while num >= v:
            result.append(s)
            num -= v
    return ''.join(result)
```

### 4.5 字符串相乘 (LeetCode 43)

```python
def multiply(num1, num2):
    if num1 == "0" or num2 == "0": return "0"
    m, n = len(num1), len(num2)
    result = [0] * (m + n)
    for i in range(m-1, -1, -1):
        for j in range(n-1, -1, -1):
            mul = (ord(num1[i])-48) * (ord(num2[j])-48)
            p1, p2 = i+j, i+j+1
            total = mul + result[p2]
            result[p2] = total % 10
            result[p1] += total // 10
    return ''.join(str(d) for d in result).lstrip('0')
```

---

## 五、复杂度分析

| 算法 | 时间复杂度 | 空间复杂度 |
|------|-----------|-----------|
| 埃氏筛 | $O(n \log \log n)$ | $O(n)$ |
| 试除法判素 | $O(\sqrt{n})$ | $O(1)$ |
| 快速幂 | $O(\log n)$ | $O(1)$ |
| GCD | $O(\log \min(a,b))$ | $O(1)$ |
| 组合数(预处理) | $O(MAXN)$ 预处理 | $O(MAXN)$ |

---

## 六、面试高频题

1. **LeetCode 7：** 整数反转
2. **LeetCode 9：** 回文数
3. **LeetCode 12/13：** 罗马数字
4. **LeetCode 43：** 字符串相乘
5. **LeetCode 50：** Pow(x, n)
6. **LeetCode 172：** 阶乘后的零
7. **LeetCode 204：** 计数质数
8. **LeetCode 264：** 丑数II
9. **LeetCode 326：** 3的幂
10. **LeetCode 372：** 超级次方
