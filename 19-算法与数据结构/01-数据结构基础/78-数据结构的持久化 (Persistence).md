## Persistence


```javascript
持久化数据结构在修改时保留历史版本，实现时间旅行。```


```
// 持久化链表节点
class PersistentNode {
  constructor(val, next=null) { this.val = val; this.next = next; }
}
class PersistentList {
  constructor() { this.versions = [null]; }
  append(version, val) {
    const newNode = new PersistentNode(val, this.versions[version]);
    this.versions.push(newNode);
    return this.versions.length - 1;
  }
  get(version) {
    const res = [];
    let cur = this.versions[version];
    while (cur) { res.push(cur.val); cur = cur.next; }
    return res;
  }
}```


  点击按钮查看结果
