## HashSet


```javascript
哈希集合基于哈希表实现，只存储键不存储值，用于快速判断元素是否存在。```


```
class HashSet {
  constructor() { this.data = {}; }
  add(x) { this.data[x] = true; }
  has(x) { return this.data[x] === true; }
  delete(x) { delete this.data[x]; }
  size() { return Object.keys(this.data).length; }
  values() { return Object.keys(this.data); }
}
const set = new HashSet();
set.add(1); set.add(2); set.add(3);
console.log(set.has(2)); // true
console.log(set.has(4)); // false```


  点击按钮查看结果
