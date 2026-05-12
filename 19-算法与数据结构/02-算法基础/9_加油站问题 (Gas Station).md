# 10-加油站问题 (Gas Station)

LeetCode 134。环形路线上有 n 个加油站，gas[i] 是加油量，cost[i] 是耗油量。找能绕行一圈的起点。

## 关键观察

1. 若总加油量 < 总耗油量，一定无解
2. 从某站出发油箱变负，起点不在该站及之前任何站
3. 贪心：从油箱变负的下一站重新开始

```javascript
function canCompleteCircuit(gas, cost) {
  let total = 0, curr = 0, start = 0;
  for (let i = 0; i < gas.length; i++) {
    total += gas[i] - cost[i];
    curr += gas[i] - cost[i];
    if (curr < 0) {
      start = i + 1; // 起点在下一站
      curr = 0;       // 重置当前油量
    }
  }
  return total >= 0 ? start : -1;
}

console.log(canCompleteCircuit([1,2,3,4,5], [3,4,5,1,2])); // 3
console.log(canCompleteCircuit([2,3,4], [3,4,3])); // -1
```

## C++ 实现

```cpp
#include <vector>
using namespace std;

int canCompleteCircuit(vector<int>& gas, vector<int>& cost) {
    int total = 0, curr = 0, start = 0;
    for (int i = 0; i < gas.size(); i++) {
        total += gas[i] - cost[i];
        curr += gas[i] - cost[i];
        if (curr < 0) {
            start = i + 1;
            curr = 0;
        }
    }
    return total >= 0 ? start : -1;
}
```

## 正确性证明

证明：若 total >= 0，则 start 是有效起点。

反证法：假设从 start 出发无法绕行一圈，那么必存在某个位置使得油量变负。但 start 本身是从前一个不可行起点的下一个位置开始的，根据贪心规则，start 之后的任何位置都不能成为更优起点。因此 start 是唯一可能的起点。

## 暴力解法（对比）

```javascript
// O(n²) 暴力：尝试每个起点
function canCompleteBruteForce(gas, cost) {
  const n = gas.length;
  for (let start = 0; start < n; start++) {
    let tank = 0;
    let canComplete = true;
    for (let i = 0; i < n; i++) {
      const station = (start + i) % n;
      tank += gas[station] - cost[station];
      if (tank < 0) { canComplete = false; break; }
    }
    if (canComplete) return start;
  }
  return -1;
}
```

## 复杂度

| 方法 | 时间 | 空间 |
|------|------|------|
| 贪心 | O(n) | O(1) |
| 暴力 | O(n²) | O(1) |

## 推广应用

- **环形缓冲区**：找到能持续写入的起始位置
- **周期任务调度**：找到能完成所有任务的起始时间
- **资源平衡**：环形资源分配中的可行性判断
- **生产者消费者**：环形队列的容量分析

## 常见陷阱

1. **total 判断**：total < 0 时一定无解
2. **start 范围**：start = i + 1 可能等于 n，需要取模
3. **单站问题**：只有一个加油站时直接判断 gas[0] >= cost[0]
4. **curr 归零**：每次重新开始时 curr 必须归零
