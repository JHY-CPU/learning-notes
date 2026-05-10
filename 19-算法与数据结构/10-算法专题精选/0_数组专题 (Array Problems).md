# 数组专题 (Array Problems)

## 一、概念定义与原理

数组是最基本的数据结构，竞赛中大量问题可以归结为数组操作。本专题介绍三种核心技巧：**双指针**、**前缀和**、**差分数组**。

### 1.1 双指针

两个指针在数组上以某种规则移动，常见模式：
- **对撞指针：** 左右两端向中间移动
- **快慢指针：** 同向移动，速度不同
- **滑动窗口：** 维护一个可变长度的区间

### 1.2 前缀和

前缀和数组 $S[i] = \sum_{j=0}^{i-1} a[j]$，$S[0] = 0$

区间和：$a[l] + a[l+1] + \cdots + a[r-1] = S[r] - S[l]$

### 1.3 差分数组

差分数组 $d[i] = a[i] - a[i-1]$，$a$ 是 $d$ 的前缀和。

区间加：$a[l \ldots r]$ 每个元素加 $v$，只需 $d[l] += v$，$d[r+1] -= v$

---

## 二、核心算法

### 2.1 两数之和（对撞指针）

给定有序数组，找两个数使和等于目标值。排序后对撞指针 $O(n)$。

### 2.2 最长无重复子串（滑动窗口）

维护窗口 $[l, r]$，保证窗口内无重复字符。右指针扩展，左指针收缩。

### 2.3 二维前缀和

$$S[i][j] = a[i][j] + S[i-1][j] + S[i][j-1] - S[i-1][j-1]$$

子矩阵和：$S[r_2][c_2] - S[r_1-1][c_2] - S[r_2][c_1-1] + S[r_1-1][c_1-1]$

---

## 三、代码实现

### 3.1 双指针 - C++

```cpp
#include <bits/stdc++.h>
using namespace std;

// 对撞指针：有序数组中找两数之和等于 target
pair<int,int> two_sum(vector<int>& a, int target) {
    int l = 0, r = a.size() - 1;
    while (l < r) {
        int sum = a[l] + a[r];
        if (sum == target) return {l, r};
        else if (sum < target) l++;
        else r--;
    }
    return {-1, -1};
}

// 快慢指针：移除有序数组中的重复元素
int remove_duplicates(vector<int>& a) {
    if (a.empty()) return 0;
    int slow = 0;
    for (int fast = 1; fast < a.size(); fast++) {
        if (a[fast] != a[slow]) a[++slow] = a[fast];
    }
    return slow + 1;
}
```

### 3.2 滑动窗口 - C++

```cpp
// 最长无重复字符子串
int length_of_longest_substring(string s) {
    unordered_map<char, int> count;
    int l = 0, result = 0;
    for (int r = 0; r < s.size(); r++) {
        count[s[r]]++;
        while (count[s[r]] > 1) {
            count[s[l]]--;
            l++;
        }
        result = max(result, r - l + 1);
    }
    return result;
}
```

### 3.3 前缀和与差分 - C++

```cpp
// 一维前缀和
vector<long long> prefix_sum(vector<int>& a) {
    int n = a.size();
    vector<long long> S(n + 1, 0);
    for (int i = 0; i < n; i++) S[i+1] = S[i] + a[i];
    return S;
}
// 区间和 query(l, r) = S[r+1] - S[l]

// 二维前缀和
vector<vector<long long>> prefix_sum_2d(vector<vector<int>>& a) {
    int m = a.size(), n = a[0].size();
    vector<vector<long long>> S(m+1, vector<long long>(n+1, 0));
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            S[i+1][j+1] = a[i][j] + S[i][j+1] + S[i+1][j] - S[i][j];
    return S;
}

// 差分数组区间修改
void range_add(vector<long long>& diff, int l, int r, long long val) {
    diff[l] += val;
    diff[r+1] -= val;
}
// 最后做前缀和还原
```

### 3.4 Python 实现

```python
def two_sum_sorted(a, target):
    l, r = 0, len(a) - 1
    while l < r:
        s = a[l] + a[r]
        if s == target: return (l, r)
        elif s < target: l += 1
        else: r -= 1
    return (-1, -1)

def prefix_sum(a):
    S = [0]
    for x in a: S.append(S[-1] + x)
    return S
# 区间和: S[r+1] - S[l]

def longest_substring(s):
    count = {}; l = 0; result = 0
    for r, c in enumerate(s):
        count[c] = count.get(c, 0) + 1
        while count[c] > 1:
            count[s[l]] -= 1; l += 1
        result = max(result, r - l + 1)
    return result

print(two_sum_sorted([1,2,3,4,5], 7))  # (1, 4)
print(longest_substring("abcabcbb"))     # 3
```

---

## 四、复杂度分析

| 技巧 | 时间复杂度 | 空间复杂度 |
|------|-----------|-----------|
| 对撞指针 | $O(n)$ | $O(1)$ |
| 滑动窗口 | $O(n)$ | $O(k)$ |
| 前缀和查询 | $O(1)$ 查询 / $O(n)$ 构建 | $O(n)$ |
| 差分修改 | $O(1)$ 修改 / $O(n)$ 还原 | $O(n)$ |
| 二维前缀和 | $O(1)$ 查询 / $O(mn)$ 构建 | $O(mn)$ |

---

## 五、竞赛与面试应用场景

1. **LeetCode 1：** 两数之和
2. **LeetCode 3：** 无重复字符的最长子串
3. **LeetCode 11：** 盛最多水的容器（对撞指针）
4. **区间求和问题：** 前缀和
5. **区间修改问题：** 差分数组
6. **子矩阵求和：** 二维前缀和
