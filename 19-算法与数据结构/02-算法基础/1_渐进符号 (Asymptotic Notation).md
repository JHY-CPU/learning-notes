# 02-渐进符号 (Asymptotic Notation)

渐进符号是描述算法运行时间随输入规模增长趋势的数学记号，忽略常数因子和低阶项。

## 核心符号

- **大O记号 O(f(n))**：上界，运行时间不超过 f(n) 的某个常数倍
- **大Omega Ω(f(n))**：下界，运行时间至少为 f(n) 的某个常数倍
- **大Theta Θ(f(n))**：紧确界，同时给出上界和下界
- **小o记号 o(f(n))**：非紧上界，增长率严格小于 f(n)
- **小omega ω(f(n))**：非紧下界，增长率严格大于 f(n)

```javascript
// 大O表示法常见复杂度排序
// O(1) < O(log n) < O(n) < O(n log n) < O(n^2) < O(2^n) < O(n!)
function compareGrowth(n) {
  return {
    constant: 1,
    logarithmic: Math.log2(n),
    linear: n,
    linearithmic: n * Math.log2(n),
    quadratic: n * n,
    exponential: Math.pow(2, Math.min(n, 20)),
  };
}
```

## C++ 验证

```cpp
#include <cmath>
#include <iostream>
using namespace std;

int main() {
    long long n = 1e6;
    cout << "O(1): " << 1 << endl;
    cout << "O(log n): " << (long long)log2(n) << endl;
    cout << "O(n): " << n << endl;
    cout << "O(n log n): " << (long long)(n * log2(n)) << endl;
    cout << "O(n^2): " << n * n << endl;
    return 0;
}
```

## 常见复杂度排序

| 复杂度 | 名称 | n=10^6 | 1秒内可处理n |
|--------|------|--------|-------------|
| O(1) | 常数 | 1 | 任意 |
| O(log n) | 对数 | 20 | 极大 |
| O(n) | 线性 | 10^6 | ~10^8 |
| O(n log n) | 线性对数 | 2×10^7 | ~10^7 |
| O(n²) | 平方 | 10^12 | ~10^4 |
| O(2^n) | 指数 | 不可行 | ~25 |
| O(n!) | 阶乘 | 不可行 | ~11 |

## 分析示例

```javascript
// O(n) - 线性
function sum(arr) {
  let s = 0;
  for (let x of arr) s += x; // 执行 n 次
  return s;
}

// O(n²) - 平方
function pairs(arr) {
  for (let i = 0; i < arr.length; i++)
    for (let j = i + 1; j < arr.length; j++) // 执行 n(n-1)/2 次
      console.log(arr[i], arr[j]);
}

// O(log n) - 对数
function pow2(n) {
  let count = 0;
  for (let i = 1; i < n; i *= 2) count++; // 执行 log2(n) 次
  return count;
}

// O(n log n) - 归并排序
// T(n) = 2T(n/2) + O(n) -> O(n log n)
```

## 在面试中的应用

1. **估算运行时间**：n=10^5 时，O(n²) 约 10^10 次操作会超时
2. **选择算法**：根据 n 的范围选择合适复杂度的算法
3. **证明最优性**：Ω(n log n) 证明排序不能更快（比较排序）
4. **分析递归**：用主定理等方法将递推式化简为渐进表达式

## 常见陷阱

1. O(2n) 和 O(n) 是同一个复杂度（忽略常数）
2. O(n + m) 不等于 O(n) 当 n 和 m 独立时
3. 最坏/平均/最好是不同概念，用 O/Ω/Θ 描述不同含义
4. 空间复杂度和时间复杂度分开分析
