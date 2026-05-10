# 滑动窗口专题 (Sliding Window)

## 一、概念定义与原理

### 1.1 核心思想

用两个指针维护一个**窗口**，窗口在数组/字符串上滑动。通过维护窗口内的状态，高效解决子数组/子串问题。

### 1.2 两种窗口

- **定长窗口：** 窗口大小固定为 $k$
- **变长窗口：** 窗口大小动态调整，满足某种条件

### 1.3 适用条件

- 数据是连续的（数组或字符串）
- 问题涉及**子数组**或**子串**
- 窗口内状态可以增量更新

---

## 二、核心算法

### 2.1 定长滑动窗口

1. 先构建初始窗口 $[0, k-1]$
2. 右指针每次移动一步，左指针同步移动
3. 增量更新窗口状态

### 2.2 变长滑动窗口

1. 右指针不断扩展
2. 当窗口不满足条件时，收缩左指针
3. 在满足条件时更新答案

### 2.3 最小覆盖子串

右指针扩展直到包含所有字符，然后收缩左指针直到刚好满足条件，记录最优解。

---

## 三、代码实现

### 3.1 定长窗口 - C++

```cpp
#include <bits/stdc++.h>
using namespace std;

// 大小为 k 的窗口的最大平均值
double find_max_average(vector<int>& nums, int k) {
    double sum = 0;
    for (int i = 0; i < k; i++) sum += nums[i];
    double result = sum;
    for (int i = k; i < nums.size(); i++) {
        sum += nums[i] - nums[i - k];
        result = max(result, sum);
    }
    return result / k;
}
```

### 3.2 变长窗口 - C++

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

// 最小覆盖子串 (LeetCode 76)
string min_window(string s, string t) {
    unordered_map<char, int> need, window;
    for (char c : t) need[c]++;
    int l = 0, matched = 0, min_len = INT_MAX, start = 0;
    for (int r = 0; r < s.size(); r++) {
        if (need.count(s[r])) {
            window[s[r]]++;
            if (window[s[r]] == need[s[r]]) matched++;
        }
        while (matched == need.size()) {
            if (r - l + 1 < min_len) {
                min_len = r - l + 1;
                start = l;
            }
            if (need.count(s[l])) {
                if (window[s[l]] == need[s[l]]) matched--;
                window[s[l]]--;
            }
            l++;
        }
    }
    return min_len == INT_MAX ? "" : s.substr(start, min_len);
}
```

### 3.3 Python 实现

```python
def max_average(nums, k):
    s = sum(nums[:k])
    result = s
    for i in range(k, len(nums)):
        s += nums[i] - nums[i-k]
        result = max(result, s)
    return result / k

def longest_substring(s):
    count = {}; l = 0; result = 0
    for r, c in enumerate(s):
        count[c] = count.get(c, 0) + 1
        while count[c] > 1:
            count[s[l]] -= 1; l += 1
        result = max(result, r - l + 1)
    return result

def min_window(s, t):
    from collections import Counter
    need = Counter(t); window = {}
    l = matched = 0; min_len = float('inf'); start = 0
    for r, c in enumerate(s):
        window[c] = window.get(c, 0) + 1
        if c in need and window[c] == need[c]: matched += 1
        while matched == len(need):
            if r - l + 1 < min_len: min_len = r - l + 1; start = l
            if s[l] in need and window[s[l]] == need[s[l]]: matched -= 1
            window[s[l]] -= 1; l += 1
    return "" if min_len == float('inf') else s[start:start+min_len]

print(longest_substring("abcabcbb"))  # 3
print(min_window("ADOBECODEBANC", "ABC"))  # "BANC"
```

### 3.4 至少包含K个重复字符的最长子串

```cpp
int longest_substring_k_repeating(string s, int k) {
    int result = 0;
    for (int unique_target = 1; unique_target <= 26; unique_target++) {
        vector<int> count(26, 0);
        int l = 0, unique = 0, at_least_k = 0;
        for (int r = 0; r < s.size(); r++) {
            if (count[s[r]-'a']++ == 0) unique++;
            if (count[s[r]-'a'] == k) at_least_k++;
            while (unique > unique_target) {
                if (count[s[l]-'a'] == k) at_least_k--;
                if (--count[s[l]-'a'] == 0) unique--;
                l++;
            }
            if (unique == unique_target && unique == at_least_k)
                result = max(result, r - l + 1);
        }
    }
    return result;
}
```

---

## 四、复杂度分析

| 问题 | 时间复杂度 | 空间复杂度 |
|------|-----------|-----------|
| 定长窗口 | $O(n)$ | $O(1)$ |
| 最长无重复子串 | $O(n)$ | $O(\text{字符集})$ |
| 最小覆盖子串 | $O(n)$ | $O(\text{字符集})$ |

---

## 五、竞赛与面试应用场景

1. **LeetCode 643：** 子数组最大平均数 I（定长窗口）
2. **LeetCode 3：** 无重复字符的最长子串
3. **LeetCode 76：** 最小覆盖子串
4. **LeetCode 438：** 找到字符串中所有字母异位词
5. **LeetCode 209：** 长度最小的子数组
