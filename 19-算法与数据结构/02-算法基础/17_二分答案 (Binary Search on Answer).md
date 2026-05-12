# 18-二分答案 (Binary Search on Answer)

二分答案是一种将二分查找思想应用到求最优解问题的技巧。当答案具有单调性（答案越大/越容易满足条件）时，可以在答案的取值范围内二分搜索，用 check 函数验证可行性。

## 核心框架

1. 确定答案的取值范围 [l, r]
2. 编写 check(mid) 函数：判断 mid 是否为可行解
3. 二分搜索，根据 check 结果缩窄范围
4. 最终得到最小可行解或最大可行解

## 单调性判断

- 若 check(x) = true 则 check(x+1) = true（单调递增），可找最小值
- 若 check(x) = true 则 check(x-1) = true（单调递减），可找最大值

## JavaScript 实现

```javascript
// 二分答案：求平方根（向下取整）
function mySqrt(x) {
  let l = 0, r = x;
  while (l <= r) {
    const mid = Math.floor((l + r) / 2);
    if (mid * mid <= x && (mid + 1) * (mid + 1) > x) return mid;
    if (mid * mid < x) l = mid + 1;
    else r = mid - 1;
  }
  return 0;
}

// 分割数组的最大值（LeetCode 410）
// 将数组分成 k 个连续子数组，最小化最大子数组和
function splitArray(nums, k) {
  let l = Math.max(...nums);
  let r = nums.reduce((a, b) => a + b, 0);

  function check(maxSum) {
    let count = 1, sum = 0;
    for (const n of nums) {
      if (sum + n > maxSum) { count++; sum = n; }
      else sum += n;
    }
    return count <= k;
  }

  while (l < r) {
    const mid = Math.floor((l + r) / 2);
    if (check(mid)) r = mid;
    else l = mid + 1;
  }
  return l;
}

// 爱吃香蕉的珂珂（LeetCode 875）
// 求最小速度 k，使得在 h 小时内吃完所有香蕉
function minEatingSpeed(piles, h) {
  let l = 1, r = Math.max(...piles);

  function canEat(speed) {
    let hours = 0;
    for (const p of piles) {
      hours += Math.ceil(p / speed);
    }
    return hours <= h;
  }

  while (l < r) {
    const mid = Math.floor((l + r) / 2);
    if (canEat(mid)) r = mid;
    else l = mid + 1;
  }
  return l;
}

// 运输能力（LeetCode 1011）
// 在 D 天内运完所有货物的最小载重
function shipWithinDays(weights, days) {
  let l = Math.max(...weights);
  let r = weights.reduce((a, b) => a + b, 0);

  function canShip(cap) {
    let d = 1, load = 0;
    for (const w of weights) {
      if (load + w > cap) { d++; load = 0; }
      load += w;
    }
    return d <= days;
  }

  while (l < r) {
    const mid = Math.floor((l + r) / 2);
    if (canShip(mid)) r = mid;
    else l = mid + 1;
  }
  return l;
}

// 测试
console.log(mySqrt(8));                          // 2
console.log(splitArray([7, 2, 5, 10, 8], 2));   // 18
console.log(minEatingSpeed([3, 6, 7, 11], 8));   // 4
console.log(shipWithinDays([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 5)); // 15
```

## C++ 实现

```cpp
#include <vector>
#include <algorithm>
#include <cmath>
using namespace std;

// 分割数组的最大值
int splitArray(vector<int>& nums, int k) {
    int l = *max_element(nums.begin(), nums.end());
    int r = 0;
    for (int n : nums) r += n;

    auto check = [&](int maxSum) {
        int count = 1, sum = 0;
        for (int n : nums) {
            if (sum + n > maxSum) { count++; sum = n; }
            else sum += n;
        }
        return count <= k;
    };

    while (l < r) {
        int mid = l + (r - l) / 2;
        if (check(mid)) r = mid;
        else l = mid + 1;
    }
    return l;
}

// 最小吃香蕉速度
int minEatingSpeed(vector<int>& piles, int h) {
    int l = 1, r = *max_element(piles.begin(), piles.end());

    auto canEat = [&](int speed) {
        long long hours = 0;
        for (int p : piles) hours += (p + speed - 1) / speed;
        return hours <= h;
    };

    while (l < r) {
        int mid = l + (r - l) / 2;
        if (canEat(mid)) r = mid;
        else l = mid + 1;
    }
    return l;
}
```

## 复杂度

| 问题 | 时间 | 空间 |
|------|------|------|
| 求平方根 | O(log n) | O(1) |
| 分割数组最大值 | O(n * log(sum)) | O(1) |
| 最小吃香蕉速度 | O(n * log(max)) | O(1) |
| 最小运输载重 | O(n * log(sum)) | O(1) |

## 常见模式

| 问题类型 | check 条件 | 搜索方向 |
|----------|-----------|----------|
| 最小化最大值 | count <= k | 找最小可行解 |
| 最大化最小值 | count >= k | 找最大可行解 |
| 恰好等于 | 满足条件 | 收缩边界 |

## 常见陷阱

1. **单调性错误**：必须确认 check 函数具有单调性才能二分
2. **边界设置**：l 至少为最大单个元素，r 至少为总和
3. **整数溢出**：mid * mid 可能溢出，用 BigInt 或转换为除法比较
4. **check 函数效率**：check 本身不能太复杂，否则总复杂度不够优

## 实际应用

二分答案是竞赛和面试中的高频技巧。关键在于将"求最优值"转化为"判断是否可行"。遇到"最小化最大值"或"最大化最小值"类问题时，优先考虑二分答案。
