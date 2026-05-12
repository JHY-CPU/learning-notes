# 12-最大子数组和 (Maximum Subarray)

Kadane 算法 O(n) 求最大子数组和，分治法 O(n log n) 也可解决。

## Kadane 算法（O(n)）

```javascript
function maxSubArray(nums) {
  let maxSum = nums[0], curSum = nums[0];
  for (let i = 1; i < nums.length; i++) {
    curSum = Math.max(nums[i], curSum + nums[i]); // 延续还是重新开始
    maxSum = Math.max(maxSum, curSum);
  }
  return maxSum;
}

// 返回子数组区间
function maxSubArrayWithIndex(nums) {
  let maxSum = nums[0], curSum = nums[0];
  let start = 0, end = 0, tempStart = 0;
  for (let i = 1; i < nums.length; i++) {
    if (nums[i] > curSum + nums[i]) {
      curSum = nums[i];
      tempStart = i;
    } else {
      curSum += nums[i];
    }
    if (curSum > maxSum) {
      maxSum = curSum;
      start = tempStart;
      end = i;
    }
  }
  return { maxSum, start, end };
}

console.log(maxSubArray([-2,1,-3,4,-1,2,1,-5,4])); // 6
```

## C++ 实现

```cpp
#include <vector>
#include <algorithm>
using namespace std;

int maxSubArray(vector<int>& nums) {
    int maxSum = nums[0], curSum = nums[0];
    for (int i = 1; i < nums.size(); i++) {
        curSum = max(nums[i], curSum + nums[i]);
        maxSum = max(maxSum, curSum);
    }
    return maxSum;
}
```

## 分治法（O(n log n)）

```javascript
function maxSubArrayDC(nums, l = 0, r = nums.length - 1) {
  if (l === r) return nums[l];
  const mid = (l + r) >> 1;

  // 左半最大
  const leftMax = maxSubArrayDC(nums, l, mid);
  // 右半最大
  const rightMax = maxSubArrayDC(nums, mid + 1, r);

  // 跨中点最大
  let leftSum = -Infinity, sum = 0;
  for (let i = mid; i >= l; i--) {
    sum += nums[i];
    leftSum = Math.max(leftSum, sum);
  }
  let rightSum = -Infinity; sum = 0;
  for (let i = mid + 1; i <= r; i++) {
    sum += nums[i];
    rightSum = Math.max(rightSum, sum);
  }

  return Math.max(leftMax, rightMax, leftSum + rightSum);
}
```

## 最大子数组乘积

```javascript
function maxProduct(nums) {
  let maxP = nums[0], minP = nums[0], result = nums[0];
  for (let i = 1; i < nums.length; i++) {
    if (nums[i] < 0) [maxP, minP] = [minP, maxP]; // 负数翻转
    maxP = Math.max(nums[i], maxP * nums[i]);
    minP = Math.min(nums[i], minP * nums[i]);
    result = Math.max(result, maxP);
  }
  return result;
}
```

## 环形最大子数组

```javascript
// 环形数组的最大子数组和 = max(普通最大, 总和 - 最小子数组和)
function maxSubarraySumCircular(nums) {
  let maxSum = nums[0], minSum = nums[0];
  let curMax = nums[0], curMin = nums[0];
  let total = nums[0];

  for (let i = 1; i < nums.length; i++) {
    curMax = Math.max(nums[i], curMax + nums[i]);
    maxSum = Math.max(maxSum, curMax);
    curMin = Math.min(nums[i], curMin + nums[i]);
    minSum = Math.min(minSum, curMin);
    total += nums[i];
  }

  // 全负数时返回最大单个元素
  if (total === minSum) return maxSum;
  return Math.max(maxSum, total - minSum);
}
```

## 复杂度

| 方法 | 时间 | 空间 |
|------|------|------|
| Kadane | O(n) | O(1) |
| 分治 | O(n log n) | O(log n) 递归栈 |
| 暴力 | O(n²) | O(1) |

## 常见陷阱

1. **全负数**：最大子数组可能只有一个元素
2. **整数溢出**：大数相加可能溢出
3. **空子数组**：题目是否允许空子数组
4. **环形数组**：需要同时考虑最小子数组
