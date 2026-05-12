# String Sliding Window

### 滑动窗口在字符串中的应用

滑动窗口是处理字符串子串/子数组问题的核心技巧。维护一个窗口 [left, right]，根据条件扩展右边界或收缩左边界，将暴力 O(n²) 优化到 O(n)。

### 关键特性

- **不定长窗口**：如最长无重复子串，窗口大小动态变化
- **定长窗口**：如长度为 k 的子串是否满足条件
- **哈希表配合**：用 Map 或数组记录窗口内字符状态
- **字符频率窗口**：维护窗口内各字符的出现次数

### 时间与空间复杂度

| 问题 | 时间复杂度 | 空间复杂度 |
|------|-----------|-----------|
| 最长无重复子串 | O(n) | O(σ) |
| 最小覆盖子串 | O(n) | O(σ) |
| 字符串排列 | O(n) | O(σ) |
| 无重复字符的最长子串 | O(n) | O(σ) |

σ 为字符集大小（如 26 或 128）。

### 适用场景 vs 替代方案

- **子串问题**：滑动窗口是首选技巧
- **子序列问题**：通常用动态规划而非滑动窗口
- **排列匹配**：固定窗口 + 频率比较
- **替代**：暴力枚举所有子串 O(n²) 不可接受

### 常见陷阱

- 混淆子串（连续）和子序列（不连续），适用不同算法
- 窗口收缩条件写错导致遗漏合法解
- 用 Map 记录频率时忘记处理计数归零的删除

```
function lengthOfLongestSubstring(s) {
  const map = new Map();
  let maxLen = 0, start = 0;
  for (let i = 0; i < s.length; i++) {
    if (map.has(s[i])) start = Math.max(start, map.get(s[i]) + 1);
    map.set(s[i], i);
    maxLen = Math.max(maxLen, i - start + 1);
  }
  return maxLen;
}
console.log(lengthOfLongestSubstring("abcabcbb")); // 3
console.log(lengthOfLongestSubstring("bbbbb")); // 1
```


### 实际应用

- **文本编辑器**：语法高亮中检测当前窗口内的关键字
- **网络协议**：TCP 滑动窗口控制数据发送速率
- **实时流处理**：在时间窗口内分析日志数据流
- **拼写检查**：在滑动窗口中检测拼写错误模式

  点击按钮查看结果
