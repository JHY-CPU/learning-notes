## Hash Table Concepts


```javascript
哈希表通过哈希函数将键映射到存储位置，实现 O(1) 平均复杂度的查找、插入和删除。```


```
class HashTable {
  constructor(size=10) {
    this.table = new Array(size).fill(null).map(() => []);
    this.size = size;
  }
  _hash(key) {
    return key.toString().split('').reduce((a,c) => a + c.charCodeAt(0), 0) % this.size;
  }
  set(key, val) {
    const idx = this._hash(key);
    const bucket = this.table[idx];
    for (const pair of bucket) {
      if (pair[0] === key) { pair[1] = val; return; }
    }
    bucket.push([key, val]);
  }
  get(key) {
    const idx = this._hash(key);
    for (const pair of this.table[idx]) {
      if (pair[0] === key) return pair[1];
    }
    return undefined;
  }
}```


  点击按钮查看结果
