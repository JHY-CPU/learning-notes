# 最长递增子序列 (LIS)

## 1. 概念与定义

最长递增子序列（Longest Increasing Subsequence，LIS）是指在一个序列中找到最长的子序列，使得子序列中的元素严格递增。子序列不要求连续。

LIS问题的核心：
- **子序列** vs **子串**：子序列不要求连续
- **递增** vs **非递减**：严格递增 vs 允许相等
- **O(n²) DP** vs **O(nlogn) 贪心+二分**

## 2. 状态定义与转移方程

### 2.1 O(n²) DP方法

```
dp[i] = 以 nums[i] 结尾的最长递增子序列长度
dp[i] = max(dp[j] + 1) for all j < i and nums[j] < nums[i]
dp[i] = 1（至少包含自己）
答案：max(dp[0], dp[1], ..., dp[n-1])
```

### 2.2 O(nlogn) 贪心+二分方法

```
tails[k] = 长度为 k+1 的递增子序列的最小末尾元素
遍历每个num：
  如果num > tails最后一个元素：追加
  否则：用num替换tails中第一个 >= num 的元素
最终答案：len(tails)
```

### 2.3 为什么贪心+二分正确

维护 tails 数组：对于每个长度 k，记录该长度下末尾元素最小的递增子序列。末尾越小，后续能接上的元素越多，这是贪心的核心。

## 3. 算法实现

### 3.1 O(n²) DP方法

```python
def lengthOfLIS_dp(nums):
    if not nums:
        return 0
    n = len(nums)
    dp = [1] * n
    for i in range(1, n):
        for j in range(i):
            if nums[j] < nums[i]:
                dp[i] = max(dp[i], dp[j] + 1)
    return max(dp)

# 同时记录路径
def lis_with_path(nums):
    n = len(nums)
    dp = [1] * n
    parent = [-1] * n
    for i in range(1, n):
        for j in range(i):
            if nums[j] < nums[i] and dp[j] + 1 > dp[i]:
                dp[i] = dp[j] + 1
                parent[i] = j

    max_len = max(dp)
    idx = dp.index(max_len)
    path = []
    while idx != -1:
        path.append(nums[idx])
        idx = parent[idx]
    path.reverse()
    return max_len, path
```

### 3.2 O(nlogn) 贪心+二分

```python
import bisect

def lengthOfLIS(nums):
    tails = []
    for num in nums:
        pos = bisect.bisect_left(tails, num)
        if pos == len(tails):
            tails.append(num)
        else:
            tails[pos] = num
    return len(tails)
```

### 3.3 最长非递减子序列

```python
def lengthOfLNDS(nums):
    """非递减：允许相等"""
    import bisect
    tails = []
    for num in nums:
        pos = bisect.bisect_right(tails, num)  # bisect_right允许相等
        if pos == len(tails):
            tails.append(num)
        else:
            tails[pos] = num
    return len(tails)
```

### 3.4 最长递减子序列

```python
def lengthOfLDS(nums):
    """最长递减子序列 = 反转数组的LIS"""
    import bisect
    tails = []
    for num in reversed(nums):
        pos = bisect.bisect_left(tails, num)
        if pos == len(tails):
            tails.append(num)
        else:
            tails[pos] = num
    return len(tails)
```

### 3.5 C++ 实现

```cpp
// O(nlogn)
int lengthOfLIS(vector<int>& nums) {
    vector<int> tails;
    for (int num : nums) {
        auto it = lower_bound(tails.begin(), tails.end(), num);
        if (it == tails.end()) tails.push_back(num);
        else *it = num;
    }
    return tails.size();
}
```

## 4. 复杂度分析

| 方法 | 时间复杂度 | 空间复杂度 |
|------|-----------|-----------|
| DP方法 | O(n²) | O(n) |
| 贪心+二分 | O(nlogn) | O(n) |
| 树状数组 | O(nlogn) | O(n) |

## 5. 典型例题

### 例题1：最长数对链（LeetCode 646）

```python
def findLongestChain(pairs):
    """按第二元素排序后贪心"""
    pairs.sort(key=lambda x: x[1])
    curr = float('-inf')
    result = 0
    for a, b in pairs:
        if a > curr:
            result += 1
            curr = b
    return result
```

### 例题2：俄罗斯套娃信封（LeetCode 354）

```python
def maxEnvelopes(envelopes):
    """二维LIS：第一维升序，第二维降序，对第二维求LIS"""
    import bisect
    envelopes.sort(key=lambda x: (x[0], -x[1]))
    tails = []
    for _, h in envelopes:
        pos = bisect.bisect_left(tails, h)
        if pos == len(tails):
            tails.append(h)
        else:
            tails[pos] = h
    return len(tails)
```

### 例题3：最长递增子序列的个数（LeetCode 673）

```python
def findNumberOfLIS(nums):
    n = len(nums)
    dp = [1] * n   # 长度
    count = [1] * n  # 方案数

    for i in range(1, n):
        for j in range(i):
            if nums[j] < nums[i]:
                if dp[j] + 1 > dp[i]:
                    dp[i] = dp[j] + 1
                    count[i] = count[j]
                elif dp[j] + 1 == dp[i]:
                    count[i] += count[j]

    max_len = max(dp)
    return sum(c for l, c in zip(dp, count) if l == max_len)
```

## 6. 常见陷阱与优化

### 6.1 常见陷阱

1. **严格递增 vs 非递减**：bisect_left（严格） vs bisect_right（非递减）
2. **贪心+二分不能还原路径**：只能得到长度
3. **最长递减子序列**：先反转数组再求LIS
4. **二维LIS**：一维排序，另一维求LIS

### 6.2 扩展问题

- **最长公共递增子序列**（LCIS）：两个序列的公共递增子序列
- **LIS的方案数**：DP方法中同时维护count数组
- **二维偏序**：第一维排序后，第二维用树状数组维护LIS
- **Dilworth定理**：最长反链 = 最小链覆盖数，LIS相关

### 6.3 Dilworth定理

```
最长递减子序列的个数 = 最少的递增子序列的覆盖数
最长递增子序列的个数 = 最少的递减子序列的覆盖数

数学形式：偏序集的最长链 = 最小反链覆盖数
```
