## String Matching Intro


```javascript
字符串匹配是在文本中查找模式串出现的位置，是字符串算法的核心问题。```


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
console.log(naiveSearch("ababcabcabababd", "ababd")); // [10]```


  点击按钮查看结果
