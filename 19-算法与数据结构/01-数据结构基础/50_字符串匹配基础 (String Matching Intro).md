# String Matching Intro

### 什么是字符串匹配

字符串匹配是在文本串（text）中查找模式串（pattern）所有出现位置的问题。是最基础的字符串算法，有多种经典解法。

### 关键特性

- **暴力匹配**：逐一尝试每个起始位置，简单但效率低
- **KMP 算法**：利用前缀表（next 数组）避免重复比较，O(n+m)
- **Boyer-Moore**：从右向左匹配，实用中最快
- **Rabin-Karp**：利用滚动哈希快速筛选候选位置

### 时间与空间复杂度

| 算法 | 时间复杂度 | 空间复杂度 | 特点 |
|------|-----------|-----------|------|
| 暴力匹配 | O(nm) | O(1) | 实现简单 |
| KMP | O(n+m) | O(m) | 理论最优 |
| Boyer-Moore | O(n/m) 最好 | O(σ) | 实际最快 |
| Rabin-Karp | O(n+m) | O(1) | 多模式匹配 |

n 为文本长度，m 为模式串长度，σ 为字符集大小。

### 适用场景 vs 替代方案

- **单模式匹配**：KMP 或内置 indexOf 即可
- **多模式匹配**：AC 自动机或 Rabin-Karp
- **模糊匹配**：编辑距离或正则表达式
- **大规模文本**：后缀数组或后缀树

### 常见陷阱

- KMP 的 next 数组构建逻辑容易出错
- 暴力匹配在长文本长模式时超时
- 忽略空模式串和模式串长于文本的边界情况

```
// 暴力匹配
function naiveSearch(text, pattern) {
  const result = [];
  for (let i = 0; i <= text.length - pattern.length; i++) {
    let match = true;
    for (let j = 0; j < pattern.length; j++) {
      if (text[i + j] !== pattern[j]) { match = false; break; }
    }
    if (match) result.push(i);
  }
  return result;
}
console.log(naiveSearch("ababcabcabababd", "ababd")); // [10]
```


### 实际应用

- **文本编辑器**：查找和替换功能的底层实现
- **搜索引擎**：在文档中定位关键词位置
- **杀毒软件**：在文件中扫描已知病毒特征码
- **DNA 分析**：在基因序列中查找特定碱基片段

  点击按钮查看结果
