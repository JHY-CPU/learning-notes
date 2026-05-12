# 08-跳跃游戏 (Jump Game)

跳跃游戏是贪心算法的经典应用，维护最远可达位置。

## Jump Game I（能否到达）

```javascript
function canJump(nums) {
  let maxReach = 0;
  for (let i = 0; i < nums.length; i++) {
    if (i > maxReach) return false; // 无法到达当前位置
    maxReach = Math.max(maxReach, i + nums[i]);
    if (maxReach >= nums.length - 1) return true;
  }
  return true;
}

console.log(canJump([2,3,1,1,4])); // true
console.log(canJump([3,2,1,0,4])); // false
```

## Jump Game II（最少跳数）

```javascript
function minJumps(nums) {
  let jumps = 0, curEnd = 0, farthest = 0;
  for (let i = 0; i < nums.length - 1; i++) {
    farthest = Math.max(farthest, i + nums[i]);
    if (i === curEnd) { // 到达当前跳跃边界
      jumps++;
      curEnd = farthest;
      if (curEnd >= nums.length - 1) break;
    }
  }
  return jumps;
}

console.log(minJumps([2,3,1,1,4])); // 2
```

## C++ 实现

```cpp
#include <vector>
#include <algorithm>
using namespace std;

bool canJump(vector<int>& nums) {
    int maxReach = 0;
    for (int i = 0; i < nums.size(); i++) {
        if (i > maxReach) return false;
        maxReach = max(maxReach, i + nums[i]);
    }
    return true;
}

int minJumps(vector<int>& nums) {
    int jumps = 0, curEnd = 0, farthest = 0;
    for (int i = 0; i < (int)nums.size() - 1; i++) {
        farthest = max(farthest, i + nums[i]);
        if (i == curEnd) {
            jumps++;
            curEnd = farthest;
        }
    }
    return jumps;
}
```

## 变种问题

```javascript
// Jump Game III（能否跳到值为0的位置）
function canReach(arr, start) {
  if (start < 0 || start >= arr.length) return false;
  if (arr[start] === 0) return true;
  if (arr[start] < 0) return false; // 已访问标记
  const jump = arr[start];
  arr[start] = -arr[start]; // 标记已访问
  return canReach(arr, start + jump) || canReach(arr, start - jump);
}

// Jump Game IV（最小步数跳到末尾，可跳到相同值）
function minJumpsIV(arr) {
  const n = arr.length;
  const map = new Map();
  for (let i = 0; i < n; i++) {
    if (!map.has(arr[i])) map.set(arr[i], []);
    map.get(arr[i]).push(i);
  }

  const queue = [0], visited = new Set([0]);
  let steps = 0;
  while (queue.length) {
    const size = queue.length;
    for (let i = 0; i < size; i++) {
      const pos = queue.shift();
      if (pos === n - 1) return steps;
      const neighbors = [pos - 1, pos + 1, ...(map.get(arr[pos]) || [])];
      for (const next of neighbors) {
        if (next >= 0 && next < n && !visited.has(next)) {
          visited.add(next);
          queue.push(next);
        }
      }
    }
    steps++;
  }
  return -1;
}
```

## 复杂度分析

| 版本 | 时间 | 空间 |
|------|------|------|
| Jump I | O(n) | O(1) |
| Jump II | O(n) | O(1) |
| Jump III | O(n) | O(n) |
| Jump IV | O(n) | O(n) |

## 关键思想

- Jump I: 维护 maxReach，检查是否能到达每个位置
- Jump II: 等价于层次 BFS，curEnd 是当前层边界
- Jump III: DFS/BFS 遍历可达位置
- Jump IV: BFS + 哈希表按值分组优化

## 常见陷阱

1. Jump II 的循环范围是 `i < n - 1`（不需要检查最后一个位置）
2. Jump III 需要标记已访问避免死循环
3. Jump IV 中相同值的位置要从 map 中移除避免重复访问
4. 所有变种都要处理空数组和单元素数组
