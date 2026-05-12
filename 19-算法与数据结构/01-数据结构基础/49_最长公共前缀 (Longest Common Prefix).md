# Longest Common Prefix

### 什么是最长公共前缀

最长公共前缀（LCP）是一组字符串中所有字符串共同拥有的最长起始子串。有多种解法：纵向扫描、横向缩减、分治、二分查找。

### 关键特性

- **纵向扫描**：逐字符比较各字符串的同一位置
- **横向缩减**：以第一个字符串为基准，逐步缩短
- **分治法**：将字符串数组分成两半，分别求 LCP 再合并
- **二分查找**：在前缀长度上二分，检查是否所有字符串都包含该前缀

### 时间与空间复杂度

| 方法 | 时间复杂度 | 空间复杂度 |
|------|-----------|-----------|
| 纵向扫描 | O(S) | O(1) |
| 横向缩减 | O(S) | O(1) |
| 分治法 | O(S) | O(m log n) |
| 二分查找 | O(S log m) | O(1) |

S 为所有字符总数，m 为最短字符串长度，n 为字符串数量。

### 适用场景 vs 替代方案

- **少量字符串**：横向缩减最直观
- **大量字符串**：分治法可并行计算
- **自动补全**：Trie 树更适合动态前缀查询

### 常见陷阱

- 空数组应返回空字符串，空字符串参与时 LCP 必为空
- 只有一个字符串时，LCP 就是该字符串本身
- 所有字符串无公共前缀时应返回 ""

```
function longestCommonPrefix(strs) {
  if (!strs.length) return '';
  let prefix = strs[0];
  for (let i = 1; i < strs.length; i++) {
    while (strs[i].indexOf(prefix) !== 0) {
      prefix = prefix.slice(0, -1);
      if (!prefix) return '';
    }
  }
  return prefix;
}
console.log(longestCommonPrefix(["flower","flow","flight"])); // fl
console.log(longestCommonPrefix(["dog","racecar","car"])); // 
```


### 实际应用

- **自动补全**：输入框提示中找出所有候选词的公共前缀
- **文件系统**：找多个路径的公共目录前缀
- **字符串压缩**：利用公共前缀减少存储
- **IP 路由**：最长前缀匹配是路由器的核心算法

  点击按钮查看结果
