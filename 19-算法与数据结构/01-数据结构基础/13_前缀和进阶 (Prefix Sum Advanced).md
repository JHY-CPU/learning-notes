# 14-前缀和进阶 (Prefix Sum Advanced)

前缀和的进阶应用，包括哈希表优化、模运算、乘积前缀等高级技巧。

## 前缀和 + 哈希表

利用哈希表存储前缀和的出现次数，解决子数组统计问题。

```javascript
// 统计和为 k 的子数组个数
function subarraySum(nums, k) {
  let map = new Map();
  map.set(0, 1);
  let count = 0, prefixSum = 0;

  for (let num of nums) {
    prefixSum += num;
    if (map.has(prefixSum - k)) {
      count += map.get(prefixSum - k);
    }
    map.set(prefixSum, (map.get(prefixSum) || 0) + 1);
  }
  return count;
}

// 和为 k 的最大子数组长度
function maxSubArrayLen(nums, k) {
  let map = new Map();
  map.set(0, -1);
  let prefixSum = 0, maxLen = 0;

  for (let i = 0; i < nums.length; i++) {
    prefixSum += nums[i];
    if (map.has(prefixSum - k)) {
      maxLen = Math.max(maxLen, i - map.get(prefixSum - k));
    }
    if (!map.has(prefixSum)) map.set(prefixSum, i);
  }
  return maxLen;
}
```

## C++ 实现

```cpp
#include <vector>
#include <unordered_map>
using namespace std;

int subarraySum(vector<int>& nums, int k) {
    unordered_map<int, int> prefix;
    prefix[0] = 1;
    int sum = 0, count = 0;
    for (int num : nums) {
        sum += num;
        if (prefix.count(sum - k)) count += prefix[sum - k];
        prefix[sum]++;
    }
    return count;
}
```

## 前缀积

```javascript
// 除自身以外数组的乘积
function productExceptSelf(nums) {
  let n = nums.length;
  let result = new Array(n).fill(1);

  let prefix = 1;
  for (let i = 0; i < n; i++) {
    result[i] = prefix;
    prefix *= nums[i];
  }

  let suffix = 1;
  for (let i = n - 1; i >= 0; i--) {
    result[i] *= suffix;
    suffix *= nums[i];
  }
  return result;
}
```

## 前缀和与模运算

```javascript
// 和可被 K 整除的子数组
function subarraysDivByK(nums, k) {
  let map = new Map();
  map.set(0, 1);
  let prefixSum = 0, count = 0;

  for (let num of nums) {
    prefixSum += num;
    // 取模处理负数
    let mod = ((prefixSum % k) + k) % k;
    if (map.has(mod)) count += map.get(mod);
    map.set(mod, (map.get(mod) || 0) + 1);
  }
  return count;
}
```

## 前缀异或

```javascript
// 异或前缀和：解决连续子数组异或问题
// 前缀异或：prefix[i] = nums[0] ^ nums[1] ^ ... ^ nums[i-1]
// 区间 [l, r] 异或 = prefix[r+1] ^ prefix[l]

function xorQueries(arr, queries) {
  let prefix = new Array(arr.length + 1).fill(0);
  for (let i = 0; i < arr.length; i++) {
    prefix[i + 1] = prefix[i] ^ arr[i];
  }
  return queries.map(([l, r]) => prefix[r + 1] ^ prefix[l]);
}

// 统计异或和为 k 的子数组个数
function countXorSubarrays(nums, k) {
  let map = new Map();
  map.set(0, 1);
  let xor = 0, count = 0;
  for (let num of nums) {
    xor ^= num;
    if (map.has(xor ^ k)) count += map.get(xor ^ k);
    map.set(xor, (map.get(xor) || 0) + 1);
  }
  return count;
}
```

## 前缀最值

```javascript
// 前缀最大值
function prefixMax(nums) {
  let prefix = [nums[0]];
  for (let i = 1; i < nums.length; i++) {
    prefix[i] = Math.max(prefix[i - 1], nums[i]);
  }
  return prefix;
}

// 前缀最小值 + 贪心
function canJump(nums) {
  let minReach = nums.length - 1;
  for (let i = nums.length - 2; i >= 0; i--) {
    if (i + nums[i] >= minReach) minReach = i;
  }
  return minReach === 0;
}
```

## 技巧总结

1. **前缀和 + 哈希表**：将子数组问题转化为前缀和差值查找
2. **前缀积用左右两遍扫描**：避免除法，处理零值
3. **模运算取正**：`((x % k) + k) % k` 处理负数
4. **前缀异或**：异或的逆运算是自身，类似前缀和

## 常见陷阱

1. 哈希表初始化 `map.set(0, 1)` 不能忘
2. 前缀积要分别从左到右和从右到左扫描
3. 模运算中负数的处理
4. 异或前缀和中初始值为 0（异或恒等元）
