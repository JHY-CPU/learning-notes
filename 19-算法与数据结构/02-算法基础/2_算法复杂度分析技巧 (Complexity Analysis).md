# 03-算法复杂度分析技巧 (Complexity Analysis)

复杂度分析不仅限于简单的循环计数，还需要掌握递归分析、嵌套循环推导、主定理套用等多种技巧。

## 递归分析方法

### 展开法

```javascript
// T(n) = T(n-1) + O(1)
// 展开：T(n) = T(n-1) + 1 = T(n-2) + 1 + 1 = ... = T(0) + n = O(n)

// T(n) = T(n-1) + O(n)
// 展开：T(n) = n + (n-1) + (n-2) + ... + 1 = n(n+1)/2 = O(n²)
```

### 树形法

```javascript
// T(n) = 2T(n/2) + O(n)
// 递归树：
// 第0层：n
// 第1层：n/2 + n/2 = n
// 第2层：n/4 * 4 = n
// ...共 log n 层
// 总代价：n * log n = O(n log n)
```

### 主定理

对于 T(n) = aT(n/b) + f(n)，设 c = log_b(a)：

```javascript
// 情况1：f(n) = O(n^(c-ε))，则 T(n) = Θ(n^c)
// 情况2：f(n) = Θ(n^c)，则 T(n) = Θ(n^c * log n)
// 情况3：f(n) = Ω(n^(c+ε))，且 af(n/b) <= cf(n)，则 T(n) = Θ(f(n))

// 应用：
// T(n) = 2T(n/2) + O(n): a=2,b=2,c=1,f(n)=n -> 情况2 -> O(n log n) 归并排序
// T(n) = T(n/2) + O(1): a=1,b=2,c=0,f(n)=1 -> 情况2 -> O(log n) 二分查找
// T(n) = 2T(n/2) + O(1): a=2,b=2,c=1,f(n)=1 -> 情况1 -> O(n) 二叉树遍历
```

## C++ 主定理应用

```cpp
// T(n) = 2T(n/2) + n → O(n log n) 归并排序
// T(n) = T(n/2) + 1 → O(log n) 二分查找
// T(n) = 2T(n/2) + 1 → O(n) 二叉树遍历
// T(n) = T(n-1) + 1 → O(n) 线性递归
// T(n) = T(n-1) + n → O(n²) 选择排序
// T(n) = 2T(n-1) + 1 → O(2^n) 朴素斐波那契
```

## 常见递推式速查

| 递推式 | 来源 | 复杂度 |
|--------|------|--------|
| T(n)=2T(n/2)+O(n) | 归并排序 | O(n log n) |
| T(n)=2T(n/2)+O(n) | 快排平均 | O(n log n) |
| T(n)=T(n/2)+O(1) | 二分查找 | O(log n) |
| T(n)=T(n-1)+O(1) | 线性递归 | O(n) |
| T(n)=T(n-1)+O(n) | 选择排序 | O(n²) |
| T(n)=2T(n-1)+O(1) | 汉诺塔 | O(2^n) |
| T(n)=T(n/2)+O(n) | 最大子数组 | O(n) |

## 嵌套循环分析

```javascript
// 1. 简单嵌套：O(n²)
for (let i = 0; i < n; i++)
  for (let j = 0; j < n; j++) // O(n²)

// 2. 内层递减：O(n²)
for (let i = 0; i < n; i++)
  for (let j = 0; j < i; j++) // 0+1+2+...+(n-1) = O(n²)

// 3. 内层倍增：O(n log n)
for (let i = 0; i < n; i++)
  for (let j = 1; j < n; j *= 2) // n * log n

// 4. 内层递减倍减：O(n log n)
for (let i = 0; i < n; i++)
  for (let j = n; j > 0; j /= 2) // n * log n

// 5. 三层循环各 n：O(n³)
for (let i = 0; i < n; i++)
  for (let j = 0; j < n; j++)
    for (let k = 0; k < n; k++) // n³
```

## 空间复杂度分析

```javascript
// O(1) 空间：原地操作
function reverse(arr) { /* 只用指针变量 */ }

// O(n) 空间：额外数组
function copy(arr) { return [...arr]; }

// O(n) 空间：递归栈
function factorial(n) {
  if (n <= 1) return 1;
  return n * factorial(n - 1); // 递归深度 n
}

// O(log n) 空间：平衡递归
function binarySearch(arr, target) {
  // 递归深度 log n
}
```

## 实用技巧

1. **先写递推式**：T(n) = 子问题代价 + 合并代价
2. **看循环变量**：i++ 是 O(n)，i*=2 是 O(log n)
3. **加法 vs 乘法**：顺序执行是加法，嵌套是乘法
4. **递归深度**：等于空间复杂度
5. **尾递归**：某些语言可优化为 O(1) 空间
