# 22-拉斯维加斯算法 (Las Vegas)

拉斯维加斯算法结果永远正确，但运行时间随机。与蒙特卡洛（时间固定、结果可能错误）相反。

## 核心特征

- 使用随机选择加速算法
- 不保证运行时间，但保证结果正确
- 典型：随机化快速排序

```javascript
// 拉斯维加斯：随机化快速排序
function randomQsort(arr) {
  if (arr.length <= 1) return arr;
  const pivot = arr[Math.floor(Math.random() * arr.length)];
  const left = [], right = [], equal = [];
  for (const x of arr) {
    if (x < pivot) left.push(x);
    else if (x > pivot) right.push(x);
    else equal.push(x);
  }
  return [...randomQsort(left), ...equal, ...randomQsort(right)];
}
```

## Monte Carlo vs Las Vegas

| 特性 | Las Vegas | Monte Carlo |
|------|-----------|-------------|
| 正确性 | 总是正确 | 可能有错 |
| 时间 | 随机 | 确定 |
| 例子 | 随机快排 | Miller-Rabin |
| 策略 | 重试直到正确 | 接受概率误差 |

## C++ 实现

```cpp
#include <vector>
#include <cstdlib>
using namespace std;

vector<int> randomQsort(vector<int> arr) {
    if (arr.size() <= 1) return arr;
    int pivot = arr[rand() % arr.size()];
    vector<int> left, right, equal;
    for (int x : arr) {
        if (x < pivot) left.push_back(x);
        else if (x > pivot) right.push_back(x);
        else equal.push_back(x);
    }
    auto l = randomQsort(left);
    auto r = randomQsort(right);
    l.insert(l.end(), equal.begin(), equal.end());
    l.insert(l.end(), r.begin(), r.end());
    return l;
}
```

## 应用场景

- 随机化快速排序（几乎所有标准库实现）
- 随机化选择算法
- 随机素性测试
- 负载均衡
- 随机化哈希函数

## 随机化选择算法（快速选择）

```javascript
// 随机化快速选择：找第 k 小元素
function randomSelect(arr, k) {
  if (arr.length === 1) return arr[0];
  const pivot = arr[Math.floor(Math.random() * arr.length)];
  const left = arr.filter(x => x < pivot);
  const equal = arr.filter(x => x === pivot);
  const right = arr.filter(x => x > pivot);
  if (k <= left.length) return randomSelect(left, k);
  if (k <= left.length + equal.length) return pivot;
  return randomSelect(right, k - left.length - equal.length);
}
// 期望 O(n)，最坏 O(n^2)
```

## 随机素性测试

```javascript
// Miller-Rabin 素性测试（Monte Carlo 方法的对偶）
// 拉斯维加斯版本：通过随机 witness 提高效率
function isPrimeLasVegas(n, trials = 20) {
  if (n < 2) return false;
  if (n < 4) return true;
  if (n % 2 === 0) return false;

  // 分解 n-1 = 2^r * d
  let d = n - 1, r = 0;
  while (d % 2 === 0) { d /= 2; r++; }

  for (let i = 0; i < trials; i++) {
    const a = 2 + Math.floor(Math.random() * (n - 3));
    let x = modPow(a, d, n);
    if (x === 1 || x === n - 1) continue;
    let found = false;
    for (let j = 0; j < r - 1; j++) {
      x = (x * x) % n;
      if (x === n - 1) { found = true; break; }
    }
    if (!found) return false;  // 一定不是素数
  }
  return true;  // 极大概率是素数
}
```

## 常见陷阱

1. **随机种子**：`srand(time(nullptr))` 只调用一次
2. **期望时间**：期望 O(n log n) 不等于总是 O(n log n)
3. **伪随机**：计算机随机是伪随机，安全性场景需用密码学随机数
4. **随机选择的偏差**：`Math.random()` 在某些场景下可能有偏差
