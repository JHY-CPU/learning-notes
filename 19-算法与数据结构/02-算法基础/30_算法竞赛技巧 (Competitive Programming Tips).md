# 31-算法竞赛技巧 (Competitive Programming Tips)

算法竞赛（OI/ACM）除了算法本身，还需要掌握大量实战技巧来应对时间限制和特殊要求。

## 时间估算

1e8 次简单操作约 1 秒（C++），根据数据规模选择算法：

| 数据规模 n | 适合的复杂度 | 典型算法 |
|-----------|-------------|---------|
| n <= 10 | O(n!) | 全排列 |
| n <= 25 | O(2^n) | 状压 DP |
| n <= 500 | O(n³) | Floyd |
| n <= 5000 | O(n²) | 区间 DP |
| n <= 1e5 | O(n log n) | 排序/线段树 |
| n <= 1e7 | O(n) | 双指针/前缀和 |
| n > 1e7 | O(log n) 或 O(1) | 二分/数学 |

## JavaScript 实现

```javascript
// 竞赛常用技巧合集

// 1. 位运算代替乘除
const double = (x) => x << 1;
const halve = (x) => x >> 1;
const isOdd = (x) => x & 1;
const isPowerOf2 = (x) => x > 0 && (x & (x - 1)) === 0;

// 2. 快速打表：预计算阶乘和组合数
const MOD = 1e9 + 7;
const MAXN = 200005;
const fact = new Array(MAXN).fill(0n);
const invFact = new Array(MAXN).fill(0n);

function initComb() {
  fact[0] = 1n;
  for (let i = 1; i < MAXN; i++) fact[i] = fact[i - 1] * BigInt(i) % BigInt(MOD);
  invFact[MAXN - 1] = modPow(fact[MAXN - 1], BigInt(MOD - 2));
  for (let i = MAXN - 2; i >= 0; i--) invFact[i] = invFact[i + 1] * BigInt(i + 1) % BigInt(MOD);
}

function modPow(base, exp) {
  let result = 1n;
  base = base % BigInt(MOD);
  while (exp > 0n) {
    if (exp & 1n) result = result * base % BigInt(MOD);
    base = base * base % BigInt(MOD);
    exp >>= 1n;
  }
  return result;
}

// 3. 快速读取（模拟 scanf）
// JavaScript 中用 readline 或 fs.readFileSync 分块读取

// 4. 差分数组模板
class DifferenceArray {
  constructor(n) {
    this.diff = new Array(n + 1).fill(0);
  }
  update(l, r, val) {
    this.diff[l] += val;
    this.diff[r + 1] -= val;
  }
  build() {
    const result = [];
    let sum = 0;
    for (let i = 0; i < this.diff.length - 1; i++) {
      sum += this.diff[i];
      result.push(sum);
    }
    return result;
  }
}

// 5. 方向数组（网格 BFS/DFS）
const dirs = [[0, 1], [0, -1], [1, 0], [-1, 0]];

// 6. 快速 GCD
function gcd(a, b) { while (b) { [a, b] = [b, a % b]; } return a; }

// 测试
console.log(double(5));       // 10
console.log(isPowerOf2(16));  // true
console.log(gcd(12, 8));      // 4
```

## C++ 实现

```cpp
#include <bits/stdc++.h>
using namespace std;

// 快读
inline int read() {
    int x = 0, f = 1;
    char ch = getchar();
    while (ch < '0' || ch > '9') { if (ch == '-') f = -1; ch = getchar(); }
    while (ch >= '0' && ch <= '9') { x = x * 10 + ch - '0'; ch = getchar(); }
    return x * f;
}

// 快写
inline void write(int x) {
    if (x < 0) { putchar('-'); x = -x; }
    if (x > 9) write(x / 10);
    putchar(x % 10 + '0');
}

// 防爆 int
typedef long long ll;
const int MOD = 1e9 + 7;

// 方向数组
int dx[] = {0, 0, 1, -1};
int dy[] = {1, -1, 0, 0};

// 最大最小值宏
#define clz(x) __builtin_clz(x)    // 前导零
#define ctz(x) __builtin_ctz(x)    // 尾随零
#define popcount __builtin_popcount // 1的个数

// 位运算技巧
bool isPowerOf2(int x) { return x > 0 && (x & (x - 1)) == 0; }
int nextPowerOf2(int x) { return 1 << (32 - __builtin_clz(x - 1)); }
```

## 防坑技巧

| 陷阱 | 解决方案 |
|------|---------|
| 浮点数相等 | 用 abs(a - b) < 1e-9 |
| 负数取模 | ((a % m) + m) % m |
| 溢出 | 用 long long / BigInt |
| 多组数据忘记清空 | 每组开始重置变量 |
| 输出格式 | 注意空格和换行 |

## 常见优化

1. **常数优化**：减少函数调用、用数组代替 map
2. **预计算**：阶乘、逆元、前缀和
3. **剪枝**：搜索中提前终止不可能的分支
4. **IO 优化**：关闭同步（C++ `ios::sync_with_stdio(false)`）

## 实际应用

拿到题先看数据规模，确定需要的复杂度级别，再选择算法。写代码时先写暴力版本，再逐步优化，确保每一步都正确。竞赛中最忌讳的是想复杂了或者实现时出错，保持代码简洁是关键。
