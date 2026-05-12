# 13-分治与主定理 (Master Theorem)

主定理求解递归式 T(n) = aT(n/b) + f(n) 的渐进复杂度。

## 三种情况

令 d = log_b(a)：

```javascript
// 情况1：f(n) = O(n^c), c < d → T(n) = Θ(n^d)
// 情况2：f(n) = Θ(n^d)       → T(n) = Θ(n^d * log n)
// 情况3：f(n) = Ω(n^c), c > d → T(n) = Θ(f(n))
```

## 典型应用

| 算法 | 递推式 | a,b,d | 结果 |
|------|--------|-------|------|
| 归并排序 | T(n)=2T(n/2)+O(n) | 2,2,1 | O(n log n) |
| 二分查找 | T(n)=T(n/2)+O(1) | 1,2,0 | O(log n) |
| 二叉树遍历 | T(n)=2T(n/2)+O(1) | 2,2,0 | O(n) |
| Strassen | T(n)=7T(n/2)+O(n^2) | 7,2,2 | O(n^2.807) |

## JavaScript 验证

```javascript
// 递归求和 T(n) = 2T(n/2) + O(1) → O(n)
function sum(arr, l, r) {
  if (l === r) return arr[l];
  const mid = (l + r) >> 1;
  return sum(arr, l, mid) + sum(arr, mid + 1, r);
}

// 二分查找 T(n) = T(n/2) + O(1) → O(log n)
function binarySearch(arr, target, l = 0, r = arr.length - 1) {
  if (l > r) return -1;
  const mid = (l + r) >> 1;
  if (arr[mid] === target) return mid;
  return arr[mid] < target
    ? binarySearch(arr, target, mid + 1, r)
    : binarySearch(arr, target, l, mid - 1);
}

// 归并排序 T(n) = 2T(n/2) + O(n) → O(n log n)
// 快排平均 T(n) = 2T(n/2) + O(n) → O(n log n)
// 快排最坏 T(n) = T(n-1) + O(n) → O(n²)
```

## 主定理推导

```
T(n) = aT(n/b) + f(n)

递归树：
- 第 0 层：f(n)
- 第 1 层：a * f(n/b)
- 第 2 层：a^2 * f(n/b^2)
- ...
- 第 k 层：a^k * f(n/b^k), k = log_b(n)

每层代价：
- 如果每层代价递减（f(n) 小）：根部主导 → 情况1
- 如果每层代价相等（f(n) = n^d）：→ 情况2 → n^d * log n
- 如果每层代价递增（f(n) 大）：叶子主导 → 情况3
```

## C++ 分析

```cpp
// T(n) = 3T(n/4) + O(n)
// a=3, b=4, d=1, log_4(3) ≈ 0.793
// d > log_b(a) → 情况3 → O(n)

// T(n) = 4T(n/2) + O(n)
// a=4, b=2, d=1, log_2(4) = 2
// d < log_b(a) → 情况1 → O(n^2)

// T(n) = T(n/2) + O(n)
// a=1, b=2, d=1, log_2(1) = 0
// d > log_b(a) → 情况3 → O(n)
```

## 局限性

主定理不适用于：
- f(n) 不是多项式形式（如 n log n）
- T(n) = aT(n/b) + f(n) 中 a, b 不是常数
- 非均匀分割（如 T(n) = T(n/3) + T(2n/3) + O(n)）

对于不适用的情况，用递归树法或展开法分析。

## 快速判断技巧

1. 写出递推式 T(n) = aT(n/b) + f(n)
2. 计算 log_b(a)
3. 比较 f(n) 和 n^{log_b(a)} 的增长速度
4. 套入三种情况之一
