# 20-随机化算法 (Randomized Algorithm)

随机化算法利用随机选择简化问题分析，避免最坏情况，将确定性最坏情况转化为概率性。

## 随机化快速排序

```javascript
function randomizedPartition(arr, l, r) {
  // 随机选择 pivot，避免最坏情况
  const ri = l + Math.floor(Math.random() * (r - l + 1));
  [arr[ri], arr[r]] = [arr[r], arr[ri]];
  return partition(arr, l, r);
}

function partition(arr, l, r) {
  const pivot = arr[r];
  let i = l - 1;
  for (let j = l; j < r; j++) {
    if (arr[j] <= pivot) {
      i++;
      [arr[i], arr[j]] = [arr[j], arr[i]];
    }
  }
  [arr[i + 1], arr[r]] = [arr[r], arr[i + 1]];
  return i + 1;
}

function randomizedQuickSort(arr, l = 0, r = arr.length - 1) {
  if (l >= r) return;
  const p = randomizedPartition(arr, l, r);
  randomizedQuickSort(arr, l, p - 1);
  randomizedQuickSort(arr, p + 1, r);
}
```

## C++ 实现

```cpp
#include <vector>
#include <cstdlib>
using namespace std;

int randomizedPartition(vector<int>& arr, int l, int r) {
    int ri = l + rand() % (r - l + 1);
    swap(arr[ri], arr[r]);
    int pivot = arr[r], i = l;
    for (int j = l; j < r; j++) {
        if (arr[j] < pivot) swap(arr[i++], arr[j]);
    }
    swap(arr[i], arr[r]);
    return i;
}

void randomizedQuickSort(vector<int>& arr, int l, int r) {
    if (l >= r) return;
    int p = randomizedPartition(arr, l, r);
    randomizedQuickSort(arr, l, p - 1);
    randomizedQuickSort(arr, p + 1, r);
}
```

## 随机化选择（第K小）

```javascript
// 期望 O(n) 找第 k 小元素
function randomizedSelect(arr, l, r, k) {
  if (l === r) return arr[l];
  const p = randomizedPartition(arr, l, r);
  const rank = p - l + 1;
  if (k === rank) return arr[p];
  if (k < rank) return randomizedSelect(arr, l, p - 1, k);
  return randomizedSelect(arr, p + 1, r, k - rank);
}

// 使用
const arr = [3, 2, 1, 5, 4];
console.log(randomizedSelect(arr, 0, 4, 2)); // 第2小 = 2
```

## Monte Carlo vs Las Vegas

| 特性 | Monte Carlo | Las Vegas |
|------|-------------|-----------|
| 结果 | 可能有错 | 总是正确 |
| 时间 | 确定 | 随机 |
| 例子 | Miller-Rabin 素数测试 | 随机化快速排序 |
| 错误概率 | 可控（2^-k） | 无 |

## 应用场景

- **快排 pivot 选择**：随机避免最坏 O(n²)
- **负载均衡**：随机分配请求到服务器
- **哈希函数**：通用哈希函数随机化
- **近似算法**：随机采样估算结果
- **博弈**：随机化策略避免被对手预测

## 复杂度分析

| 算法 | 期望 | 最坏 |
|------|------|------|
| 随机快排 | O(n log n) | O(n²) |
| 随机选择 | O(n) | O(n²) |
| 随机采样 | O(k) | O(n) |

期望复杂度通过概率分析得到：随机选择 pivot 使期望递归深度为 O(log n)。

## 常见陷阱

1. **随机种子**：不要在循环内重复调用 `srand`
2. **概率误解**：期望 O(n) 不等于总是 O(n)
3. **伪随机**：计算机中的随机是伪随机
4. **公平性**：确保每个元素被选中的概率相等
