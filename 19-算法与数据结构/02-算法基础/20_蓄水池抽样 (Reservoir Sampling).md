# 21-蓄水池抽样 (Reservoir Sampling)

蓄水池抽样用于在不知道数据总量（数据流）的情况下，等概率地抽取 k 个样本。

## 算法步骤

```javascript
function reservoirSample(stream, k) {
  const reservoir = [];
  for (let i = 0; i < k && i < stream.length; i++) {
    reservoir.push(stream[i]);
  }
  for (let i = k; i < stream.length; i++) {
    const j = Math.floor(Math.random() * (i + 1));
    if (j < k) reservoir[j] = stream[i];
  }
  return reservoir;
}

// 测试
const stream = [1,2,3,4,5,6,7,8,9,10];
console.log(reservoirSample(stream, 3)); // 每次结果不同
```

## C++ 实现

```cpp
#include <vector>
#include <cstdlib>
#include <ctime>
using namespace std;

vector<int> reservoirSample(vector<int>& stream, int k) {
    srand(time(nullptr));
    vector<int> reservoir(k);
    for (int i = 0; i < k && i < stream.size(); i++)
        reservoir[i] = stream[i];
    for (int i = k; i < stream.size(); i++) {
        int j = rand() % (i + 1);
        if (j < k) reservoir[j] = stream[i];
    }
    return reservoir;
}
```

## 正确性证明

对于第 i 个元素被选入蓄水池的概率 = k/i。之后每一步它被保留的概率为 (1 - k/(j+1) * 1/k) = j/(j+1)。

乘积化简：k/i * i/(i+1) * (i+1)/(i+2) * ... * (n-1)/n = k/n

因此每个元素最终被选中的概率都是 k/n。

## k = 1 的特殊情况

```javascript
// 从数据流中随机选一个元素
function randomPick(stream) {
  let result = null;
  for (let i = 0; i < stream.length; i++) {
    if (Math.random() < 1 / (i + 1)) result = stream[i];
  }
  return result;
}
```

## 链表随机节点（LeetCode 382）

```javascript
class Solution {
  constructor(head) { this.head = head; }

  getRandom() {
    let result = null, count = 0;
    let curr = this.head;
    while (curr) {
      count++;
      if (Math.random() < 1 / count) result = curr.val;
      curr = curr.next;
    }
    return result;
  }
}
```

## 复杂度

| 操作 | 时间 | 空间 |
|------|------|------|
| 抽样 | O(n) | O(k) |

## 应用场景

- 随机抽取数据库记录
- 日志系统随机采样
- 搜索结果随机化
- A/B 测试用户分组
- 大数据随机样本

## 常见陷阱

1. **随机数范围**：`rand() % (i + 1)` 而非 `rand() % i`
2. **k 超过 n**：当 stream 长度小于 k 时直接返回全部
3. **概率计算**：理解每个元素被选中的概率确实相等
4. **权重不等**：加权蓄水池抽样需要不同处理
