# Hash Table Counting

### 哈希表在计数统计中的应用

哈希表是频率统计的最佳工具，用键存储待统计的元素，值存储出现次数。一次遍历即可完成统计，时间复杂度 O(n)。

### 关键特性

- **频率统计**：遍历数据，每遇到一个元素就将对应计数加一
- **多数元素**：利用哈希表计数后筛选，或 Boyer-Moore 投票法 O(1) 空间
- **Top-K 问题**：哈希计数 + 堆选择，O(n log k) 时间

### 时间与空间复杂度

| 问题 | 时间复杂度 | 空间复杂度 |
|------|-----------|-----------|
| 频率统计 | O(n) | O(n) |
| 多数元素 | O(n) | O(n) |
| Top-K 频率 | O(n log k) | O(n) |
| 异位词判断 | O(n) | O(26) |

### 适用场景 vs 替代方案

- **频率统计**：哈希表 vs 排序后扫描（O(n) vs O(n log n)）
- **多数元素**：哈希表 vs Boyer-Moore 投票法（O(n) 空间 vs O(1)）
- **有限字符集**：用固定数组替代哈希表（如 26 个字母）

### 常见陷阱

- 计数时未初始化默认值为 0，导致 undefined + 1 = NaN
- 遍历 Map 时使用 forEach 比 for...of 更安全
- 大量不同元素时哈希表空间开销可能很大

```
// 多数元素
function majorityElement(nums) {
  const count = new Map();
  for (const n of nums) count.set(n, (count.get(n) || 0) + 1);
  for (const [n, c] of count) if (c > nums.length / 2) return n;
  return null;
}
// 找出所有出现次数 > n/3 的元素
function majorityElement2(nums) {
  const count = new Map();
  for (const n of nums) count.set(n, (count.get(n) || 0) + 1);
  const res = [];
  for (const [n, c] of count) if (c > Math.floor(nums.length / 3)) res.push(n);
  return res;
}
console.log(majorityElement([1,2,3,2,2,2,5,4,2])); // 2
```


### 实际应用

- **搜索引擎**：统计词频用于 TF-IDF 排序
- **日志分析**：统计错误码出现次数，快速定位高频问题
- **推荐系统**：统计用户行为频率，构建兴趣画像
- **生物信息学**：DNA 序列中碱基出现频率分析

  点击按钮查看结果
