# 09-分发饼干 (Assign Cookies)

LeetCode 455。贪心地满足最多孩子：给每个孩子分配不小于其胃口的最小饼干。

## 算法

```javascript
function findContentChildren(g, s) {
  g.sort((a, b) => a - b); // 孩子胃口升序
  s.sort((a, b) => a - b); // 饼干大小升序
  let child = 0, cookie = 0;
  while (child < g.length && cookie < s.length) {
    if (s[cookie] >= g[child]) child++; // 饼干满足孩子
    cookie++; // 无论是否满足，饼干都用掉了
  }
  return child;
}

console.log(findContentChildren([1, 2, 3], [1, 1])); // 1
console.log(findContentChildren([1, 2], [1, 2, 3])); // 2
```

## C++ 实现

```cpp
#include <vector>
#include <algorithm>
using namespace std;

int findContentChildren(vector<int>& g, vector<int>& s) {
    sort(g.begin(), g.end());
    sort(s.begin(), s.end());
    int child = 0, cookie = 0;
    while (child < g.size() && cookie < s.size()) {
        if (s[cookie] >= g[child]) child++;
        cookie++;
    }
    return child;
}
```

## 正确性证明

贪心选择性质：用最小的能满足当前孩子的饼干是最优的。因为：
- 如果用更大的饼干满足胃口小的孩子，可能使胃口大的孩子无法满足
- 用最小可行饼干保留了更多大饼干给后面胃口大的孩子

## 变种：分发糖果

```javascript
// LeetCode 135: 分发糖果
// 每个孩子至少1个，评分高于邻居的孩子获得更多糖果
function candy(ratings) {
  const n = ratings.length;
  const candies = new Array(n).fill(1);

  // 从左到右
  for (let i = 1; i < n; i++) {
    if (ratings[i] > ratings[i - 1]) candies[i] = candies[i - 1] + 1;
  }

  // 从右到左
  for (let i = n - 2; i >= 0; i--) {
    if (ratings[i] > ratings[i + 1]) {
      candies[i] = Math.max(candies[i], candies[i + 1] + 1);
    }
  }

  return candies.reduce((a, b) => a + b, 0);
}
```

## 复杂度

| 操作 | 时间 | 空间 |
|------|------|------|
| 排序 | O(n log n) | O(1) |
| 双指针 | O(n) | O(1) |
| 总计 | O(n log n) | O(1) |

## 推广

这类问题的通用模式：有限资源分配，最大化满足数。类似场景：
- 任务分配给工人
- 频段分配给用户
- 座位分配给乘客
- 服务器容量分配给请求

## 常见陷阱

1. **排序方向**：孩子和饼干都要升序
2. **指针移动**：cookie 指针每次都要移动（无论是否满足）
3. **空数组**：孩子或饼干为空时返回 0
4. **贪心方向**：不要用大饼干满足小胃口的孩子
