# DP经典问题汇总 (Classic DP Problems)

## 1. 线性DP

### 1.1 基础问题

| 问题 | LeetCode | 核心思想 |
|------|---------|---------|
| 爬楼梯 | 70 | 斐波那契 |
| 打家劫舍 | 198/213/337 | 线性/环形/树形 |
| 最大子数组和 | 53 | Kadane算法 |
| 最长递增子序列 | 300 | O(n²)DP / O(nlogn)贪心+二分 |
| 零钱兑换 | 322 | 完全背包 |
| 单词拆分 | 139 | 完全背包 |

### 1.2 代码示例：最大子数组和

```python
def maxSubArray(nums):
    max_sum = curr = nums[0]
    for x in nums[1:]:
        curr = max(x, curr + x)
        max_sum = max(max_sum, curr)
    return max_sum
```

## 2. 背包DP

### 2.1 经典背包

| 问题 | 类型 | 关键 |
|------|------|------|
| 0-1背包 | 最值 | 逆序遍历 |
| 完全背包 | 最值 | 正序遍历 |
| 多重背包 | 最值 | 二进制拆分 |
| 目标和 | 0-1背包方案数 | 转化为背包 |
| 零钱兑换II | 完全背包方案数 | 组合vs排列 |

### 2.2 代码示例：目标和

```python
def findTargetSumWays(nums, target):
    total = sum(nums)
    if (target + total) % 2: return 0
    cap = (target + total) // 2
    dp = [0] * (cap + 1)
    dp[0] = 1
    for num in nums:
        for j in range(cap, num - 1, -1):
            dp[j] += dp[j - num]
    return dp[cap]
```

## 3. 区间DP

### 3.1 经典区间DP

| 问题 | 核心思想 |
|------|---------|
| 石子合并 | 枚举分割点 |
| 矩阵链乘法 | 枚举分割点 |
| 戳气球 | 逆向思考 |
| 最长回文子序列 | 首尾比较 |

### 3.2 代码示例：戳气球

```python
def maxCoins(nums):
    nums = [1] + nums + [1]
    n = len(nums)
    dp = [[0] * n for _ in range(n)]
    for length in range(2, n):
        for left in range(n - length):
            right = left + length
            for k in range(left + 1, right):
                dp[left][right] = max(dp[left][right],
                    dp[left][k] + dp[k][right] + nums[left]*nums[k]*nums[right])
    return dp[0][n-1]
```

## 4. 网格DP

| 问题 | LeetCode | 核心 |
|------|---------|------|
| 不同路径 | 62 | 路径计数 |
| 最小路径和 | 64 | 路径最值 |
| 最大正方形 | 221 | 三方取min |
| 地下城游戏 | 174 | 反向DP |

## 5. 字符串DP

| 问题 | LeetCode | 核心 |
|------|---------|------|
| 最长公共子序列 | 1143 | 二维DP |
| 编辑距离 | 72 | 三种操作 |
| 正则表达式匹配 | 10 | 处理.* |
| 最长回文子串 | 5 | 区间DP/中心扩展 |

## 6. 树形DP

| 问题 | LeetCode | 核心 |
|------|---------|------|
| 打家劫舍III | 337 | 后序遍历 |
| 树的直径 | 543 | 全局变量 |
| 监控二叉树 | 968 | 三状态 |

## 7. 状态压缩DP

| 问题 | 核心 |
|------|------|
| 旅行商问题 | 位掩码表示访问集合 |
| 最短Hamilton路径 | dp[mask][last] |

## 8. 数位DP

| 问题 | 核心 |
|------|------|
| 统计不含某数字的数 | 记忆化搜索+limit |
| 各位数字互不相同的数 | 位掩码记录已用数字 |

## 9. 博弈DP

| 问题 | LeetCode | 核心 |
|------|---------|------|
| Nim游戏 | 292 | 异或和 |
| 石子游戏 | 877 | 区间博弈 |

## 10. 总结

```
一维DP：线性递推，如爬楼梯、打家劫舍
二维DP：网格/双串，如路径、LCS、编辑距离
背包DP：容量约束，0-1/完全/多重
区间DP：合并操作，枚举分割点
树形DP：后序遍历，父子转移
状压DP：位运算，n <= 20
数位DP：逐位构造，记忆化搜索
博弈DP：最优决策，双方对抗
```
