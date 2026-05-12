# 35-位运算在算法中的应用 (Bitwise Applications)

位运算是直接操作二进制位的运算，能实现常数优化和状态压缩。

## 常用位运算

| 操作 | 符号 | 用途 |
|------|------|------|
| 按位与 | & | 掩码、判断奇偶、lowbit |
| 按位或 | \| | 合并状态、设置位 |
| 按位异或 | ^ | 交换、找不同元素 |
| 左移 | << | 乘以 2^n |
| 右移 | >> | 除以 2^n |
| 取反 | ~ | 位翻转 |

## 核心技巧

```javascript
// 快速幂 O(log n)
function fastPow(base, exp, mod = 1e9 + 7) {
  let result = 1;
  base %= mod;
  while (exp > 0) {
    if (exp & 1) result = (result * base) % mod;
    base = (base * base) % mod;
    exp >>= 1;
  }
  return result;
}

// lowbit：取最低位的 1
function lowbit(x) { return x & (-x); }

// 统计 1 的个数（popcount）
function popcount(x) {
  let c = 0;
  while (x) { c++; x &= x - 1; } // 消除最低位 1
  return c;
}

// 判断是否是 2 的幂
function isPowerOfTwo(n) { return n > 0 && (n & (n - 1)) === 0; }

// 不用临时变量交换
function swap(a, b) {
  a ^= b; b ^= a; a ^= b;
  return [a, b];
}

// 找只出现一次的数（其他都出现两次）
function singleNumber(nums) {
  return nums.reduce((a, b) => a ^ b, 0);
}

// 找两个只出现一次的数
function singleNumberTwo(nums) {
  let xor = nums.reduce((a, b) => a ^ b, 0);
  const bit = xor & (-xor); // 取最右的 1
  let a = 0, b = 0;
  for (const n of nums) {
    if (n & bit) a ^= n;
    else b ^= n;
  }
  return [a, b];
}
```

## C++ 实现

```cpp
#include <vector>
using namespace std;

// 快速幂
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

// 位运算技巧
int lowbit(int x) { return x & (-x); }
int popcount(int x) { return __builtin_popcount(x); }
bool isPowerOfTwo(int n) { return n > 0 && (n & (n-1)) == 0; }
int singleNumber(vector<int>& nums) {
    int res = 0;
    for (int n : nums) res ^= n;
    return res;
}
```

## 子集枚举（状态压缩）

```javascript
// 用二进制位表示子集
function enumerateSubsets(arr) {
  const n = arr.length;
  const result = [];
  for (let mask = 0; mask < (1 << n); mask++) {
    const subset = [];
    for (let i = 0; i < n; i++) {
      if (mask & (1 << i)) subset.push(arr[i]);
    }
    result.push(subset);
  }
  return result;
}

console.log(enumerateSubsets([1, 2, 3]));
// [[],[1],[2],[1,2],[3],[1,3],[2,3],[1,2,3]]
```

## 位运算 DP

```javascript
// 旅行商问题（状态压缩 DP）
function tsp(dist) {
  const n = dist.length;
  const dp = Array.from({length: 1 << n}, () => new Array(n).fill(Infinity));
  dp[1][0] = 0; // 从城市0出发

  for (let mask = 1; mask < (1 << n); mask++) {
    for (let u = 0; u < n; u++) {
      if (!(mask & (1 << u))) continue;
      for (let v = 0; v < n; v++) {
        if (mask & (1 << v)) continue;
        const newMask = mask | (1 << v);
        dp[newMask][v] = Math.min(dp[newMask][v], dp[mask][u] + dist[u][v]);
      }
    }
  }

  let result = Infinity;
  const fullMask = (1 << n) - 1;
  for (let i = 0; i < n; i++) {
    result = Math.min(result, dp[fullMask][i] + dist[i][0]);
  }
  return result;
}
```

## 应用场景

- 快速幂取模（密码学）
- 状态压缩 DP（TSP、集合覆盖）
- 子集枚举
- 位掩码权限管理
- 异或找不同元素
- 位图（Bitmap）

## 常见陷阱

1. **优先级**：位运算优先级低于比较运算，要加括号
2. **负数右移**：有符号右移是算术右移（补符号位）
3. **移位越界**：左移超过类型位数是未定义行为
4. **整数溢出**：位运算可能产生溢出
