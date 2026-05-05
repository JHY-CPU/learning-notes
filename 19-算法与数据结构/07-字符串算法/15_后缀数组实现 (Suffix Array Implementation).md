## Suffix Array Implementation


```javascript
后缀数组实现及其应用：最长公共前缀（LCP）、子串查询等。```


```
// 后缀数组（简化实现）
function buildSuffixArray(s) {
  const n = s.length;
  const suffixes = [];
  for (let i = 0; i < n; i++) suffixes.push({index: i, suffix: s.slice(i)});
  suffixes.sort((a,b) => a.suffix.localeCompare(b.suffix));
  return suffixes.map(s => s.index);
}
// LCP 数组（Kasai 算法）
function buildLCP(s, sa) {
  const n = s.length;
  const rank = new Array(n);
  for (let i = 0; i < n; i++) rank[sa[i]] = i;
  let k = 0;
  const lcp = new Array(n-1).fill(0);
  for (let i = 0; i < n; i++) {
    if (rank[i] === n-1) { k = 0; continue; }
    let j = sa[rank[i] + 1];
    while (i + k < n && j + k < n && s[i+k] === s[j+k]) k++;
    lcp[rank[i]] = k;
    if (k) k--;
  }
  return lcp;
}
const s = "banana";
const sa = buildSuffixArray(s);
console.log(sa); // [5,3,1,0,4,2] (a,ana,anana,banana,na,nana)```


  点击按钮查看结果
