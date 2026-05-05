## Offline Algorithm


```javascript
离线算法预先知道所有输入后统一处理，通常能取得更好的全局解。```


```
// 离线查询：区间和（前缀和）
class OfflineRangeSum {
  constructor(arr) {
    this.prefix = [0];
    for (const x of arr) this.prefix.push(this.prefix[this.prefix.length-1] + x);
  }
  query(l, r) { return this.prefix[r+1] - this.prefix[l]; }
}
// 离线查询：莫队算法（区间计数）
console.log('离线算法可以先排序查询，统一处理');```


  点击按钮查看结果
