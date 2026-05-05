## Bitwise Sets


```javascript
用二进制位表示集合，可以实现 O(1) 的插入、删除、判断和遍历。```


```
// 用整数表示集合（最多32个元素）
const universe = ['A','B','C','D','E'];
let set = 0;
function add(s, i) { return s | (1 << i); }
function remove(s, i) { return s & ~(1 << i); }
function has(s, i) { return (s & (1 << i)) !== 0; }
function toArray(s) { return universe.filter((_,i) => has(s,i)); }
set = add(set, 0); // A
set = add(set, 2); // C
set = add(set, 4); // E
console.log(toArray(set)); // ['A','C','E']
set = remove(set, 2);
console.log(toArray(set)); // ['A','E']```


  点击按钮查看结果
