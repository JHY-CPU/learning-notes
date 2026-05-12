# 29-算法优化技巧 (Algorithm Optimization)

在正确算法的基础上，通过各种技巧进一步提升效率。

## 常用优化技巧

```javascript
// 1. 剪枝：提前终止不可能产生最优解的搜索分支
function backtrackWithPruning(state, best) {
  if (bound(state) >= best) return; // 剪枝
  // 继续搜索...
}

// 2. 记忆化：缓存子问题结果
const memo = new Map();
function fibMemo(n) {
  if (n <= 1) return n;
  if (memo.has(n)) return memo.get(n);
  const result = fibMemo(n-1) + fibMemo(n-2);
  memo.set(n, result);
  return result;
}

// 3. 滚动数组：DP 中只保留最近几行
function dpRolling(n) {
  let prev = new Array(n).fill(0);
  let curr = new Array(n).fill(0);
  for (let i = 1; i <= n; i++) {
    [prev, curr] = [curr, prev]; // 滚动
    for (let j = 0; j < n; j++) {
      curr[j] = /* 状态转移 */;
    }
  }
  return curr[n-1];
}

// 4. 预计算：提前计算常用值
function precomputePrimes(n) {
  const isPrime = new Array(n + 1).fill(true);
  isPrime[0] = isPrime[1] = false;
  for (let i = 2; i * i <= n; i++) {
    if (isPrime[i]) for (let j = i * i; j <= n; j += i) isPrime[j] = false;
  }
  return isPrime;
}
```

## C++ 优化示例

```cpp
#include <vector>
using namespace std;

// 位运算优化
int countBits(int x) {
    return __builtin_popcount(x);
}

// 空间优化：原地操作
void reverseInPlace(vector<int>& arr) {
    int l = 0, r = arr.size() - 1;
    while (l < r) swap(arr[l++], arr[r--]);
}

// 常数优化：减少函数调用
int sumFast(vector<int>& arr) {
    int s = 0;
    int n = arr.size();
    int* p = arr.data();
    for (int i = 0; i < n; i++) s += p[i]; // 直接指针访问
    return s;
}
```

## 空间换时间

| 技巧 | 空间 | 收益 |
|------|------|------|
| 哈希表 | O(n) | O(1) 查找 |
| 前缀和 | O(n) | O(1) 区间查询 |
| 预处理表 | O(n^2) | O(1) 查询 |
| 记忆化 | O(n) | 避免重复计算 |

## 时间换空间

| 技巧 | 时间代价 | 空间收益 |
|------|---------|---------|
| 滚动数组 | 无 | O(n) -> O(1) |
| 原地操作 | 无 | O(n) -> O(1) |
| 重新计算 | O(n) | O(n) -> O(1) |

## 优化顺序

1. **算法层面**：O(n^2) -> O(n log n) -> O(n)
2. **数据结构**：数组 -> 哈希表 -> 堆
3. **常数优化**：减少循环次数、避免重复计算
4. **底层优化**：缓存友好、位运算

## 常见陷阱

1. **过度优化**：正确性比效率更重要
2. **优化错误方向**：先确认瓶颈在哪里
3. **牺牲可读性**：极端优化可能使代码难以维护
4. **忽略大O**：常数优化在 n 很大时效果有限
