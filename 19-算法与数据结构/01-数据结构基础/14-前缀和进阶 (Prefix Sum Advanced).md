## 14-前缀和进阶 (Prefix Sum Advanced)

前缀和的进阶应用，包括哈希表优化、模运算、乘积前缀等高级技巧。

## 前缀和 + 哈希表

利用哈希表存储前缀和的出现次数，解决子数组统计问题。

```javascript

// 问题：统计和为 k 的子数组个数
function subarraySum(nums, k) {
  let map = new Map();
  map.set(0, 1); // 前缀和为0出现1次
  let count = 0;
  let prefixSum = 0;

  for (let num of nums) {
    prefixSum += num;

    // 如果存在 prefixSum - k 的前缀和，说明有子数组和为k
    if (map.has(prefixSum - k)) {
      count += map.get(prefixSum - k);
    }

    // 记录当前前缀和
    map.set(prefixSum, (map.get(prefixSum) || 0) + 1);
  }

  return count;
}

// 问题：和为 k 的最大子数组长度
function maxSubArrayLen(nums, k) {
  let map = new Map(); // 前缀和 -> 最早出现索引
  map.set(0, -1);
  let prefixSum = 0;
  let maxLen = 0;

  for (let i = 0; i < nums.length; i++) {
    prefixSum += nums[i];
    // 我们希望找到 prefixSum - k 的最早出现位置
    if (map.has(prefixSum - k)) {
      maxLen = Math.max(maxLen, i - map.get(prefixSum - k));
    }
    // 只存最早出现的位置
    if (!map.has(prefixSum)) {
      map.set(prefixSum, i);
    }
  }

  return maxLen;
}
```

## 前缀积

```javascript

// 前缀积（处理0的边界情况）
function productExceptSelf(nums) {
  let n = nums.length;
  let result = new Array(n).fill(1);

  // 从左到右的前缀积
  let prefix = 1;
  for (let i = 0; i < n; i++) {
    result[i] = prefix;
    prefix *= nums[i];
  }

  // 从右到左的后缀积
  let suffix = 1;
  for (let i = n - 1; i >= 0; i--) {
    result[i] *= suffix;
    suffix *= nums[i];
  }

  return result; // 每个位置是除了自身外所有元素的乘积
}
```

## 交互演示
