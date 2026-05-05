## Hash Techniques


```javascript
哈希技巧包括字符串哈希、滚动哈希、双哈希、一致性哈希。```


```
// 一致性哈希（简化版）
class ConsistentHash {
  constructor(nodes, replicas=3) {
    this.replicas = replicas;
    this.ring = {};
    this.keys = [];
    for (const node of nodes) this.addNode(node);
  }
  addNode(node) {
    for (let i = 0; i < this.replicas; i++) {
      const hash = this._hash(`${node}:${i}`);
      this.ring[hash] = node;
      this.keys.push(hash);
    }
    this.keys.sort((a,b) => a-b);
  }
  _hash(s) { let h=0; for (const c of s) h = (h*31 + c.charCodeAt(0)) % 10000; return h; }
  get(key) {
    if (!this.keys.length) return null;
    const hash = this._hash(key);
    for (const k of this.keys) if (k >= hash) return this.ring[k];
    return this.ring[this.keys[0]];
  }
}```


  点击按钮查看结果
