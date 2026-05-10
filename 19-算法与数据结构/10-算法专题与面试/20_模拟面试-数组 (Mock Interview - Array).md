# 模拟面试-数组 (Mock Interview - Array)

## 一、面试流程模拟

**时间：** 45分钟
**形式：** 面试官提问，候选人口述思路 + 编码

---

## 二、题目1：两数之和 (LeetCode 1, Easy, 10分钟)

### 题目描述

给定整数数组 `nums` 和目标值 `target`，找出和为 `target` 的两个整数的下标。

### 面试过程

**面试官：** "请先描述一下你的思路。"

**候选人：**
"最直接的思路是双重循环，对每个元素查找另一个元素，时间 O(n^2)。

但我们可以用哈希表优化。遍历时，对每个数检查 `target - num` 是否已经在哈希表中。如果在，找到答案；如果不在，把当前数和下标存入哈希表。这样只需一次遍历，时间 O(n)，空间 O(n)。"

### 代码

```python
def two_sum(nums, target):
    seen = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return []
```

**面试官追问：** "如果有重复元素怎么办？"
"不影响，哈希表只存第一个遇到的下标。但如果有多个答案，这个方法只返回第一个找到的。"

---

## 三、题目2：接雨水 (LeetCode 42, Hard, 25分钟)

### 题目描述

给定 `n` 个非负整数表示宽度为1的柱子的高度图，计算下雨后能接多少水。

### 面试过程

**候选人：**
"每个位置能接的水 = min(左侧最高, 右侧最高) - 当前高度，如果为正的话。

有几种方法：
1. **暴力：** 对每个位置向左向右找最高，O(n^2)
2. **DP预处理：** 预计算每个位置的左右最高，O(n) 时间 O(n) 空间
3. **双指针：** 从两端向中间，O(n) 时间 O(1) 空间

我用双指针方法。"

### 代码

```python
def trap(height):
    left, right = 0, len(height) - 1
    lmax = rmax = water = 0

    while left < right:
        lmax = max(lmax, height[left])
        rmax = max(rmax, height[right])
        if lmax < rmax:
            water += lmax - height[left]
            left += 1
        else:
            water += rmax - height[right]
            right -= 1

    return water
```

**面试官追问：** "为什么当 lmax < rmax 时可以确定左边的水量？"
"因为此时右侧已经有一个 >= lmax 的值，所以左边这个位置的盛水量就受限于 lmax。即使右边还有更高的柱子，也不影响当前结果。"

---

## 四、题目3：三数之和 (LeetCode 15, Medium, 15分钟)

### 题目描述

找出数组中所有和为0的三元组（不能包含重复）。

### 面试过程

**候选人：**
"排序 + 双指针。先排序，然后固定第一个数，用双指针找另外两个数。跳过重复元素避免重复三元组。"

### 代码

```python
def three_sum(nums):
    nums.sort()
    result = []
    for i in range(len(nums) - 2):
        if i > 0 and nums[i] == nums[i-1]:
            continue
        left, right = i + 1, len(nums) - 1
        while left < right:
            total = nums[i] + nums[left] + nums[right]
            if total == 0:
                result.append([nums[i], nums[left], nums[right]])
                while left < right and nums[left] == nums[left+1]: left += 1
                while left < right and nums[right] == nums[right-1]: right -= 1
                left += 1; right -= 1
            elif total < 0: left += 1
            else: right -= 1
    return result
```

**复杂度：** 时间 $O(n^2)$，排序 $O(n \log n)$ + 双指针 $O(n^2)$。空间 $O(1)$（不计输出）。

---

## 五、评分标准

| 评价维度 | 优秀 | 及格 | 不足 |
|---------|------|------|------|
| 思路清晰度 | 能准确说出多种方法及优劣 | 能说出最优方法 | 说不出方法 |
| 编码能力 | 快速写出正确代码 | 最终能写对 | 频繁出错 |
| 复杂度分析 | 主动说明并对比 | 被问后能答出 | 答不出 |
| 沟通能力 | 边写边解释 | 有回应 | 沉默编码 |
