## Skip List


```javascript
跳表在有序链表上增加多层索引，实现 O(log n) 的平均查找复杂度。```


```
class SkipNode {
  constructor(val, level) {
    this.val = val;
    this.next = new Array(level + 1).fill(null);
  }
}
class SkipList {
  constructor(maxLevel=16) { this.maxLevel = maxLevel; this.head = new SkipNode(-Infinity, maxLevel); this.level = 0; }
  _randomLevel() { let l = 0; while (Math.random() < 0.5 && l < this.maxLevel) l++; return l; }
  search(target) {
    let cur = this.head;
    for (let i = this.level; i >= 0; i--) {
      while (cur.next[i] && cur.next[i].val < target) cur = cur.next[i];
    }
    cur = cur.next[0];
    return cur && cur.val === target;
  }
  insert(num) {
    const update = new Array(this.maxLevel + 1).fill(null);
    let cur = this.head;
    for (let i = this.level; i >= 0; i--) {
      while (cur.next[i] && cur.next[i].val < num) cur = cur.next[i];
      update[i] = cur;
    }
    const lvl = this._randomLevel();
    if (lvl > this.level) {
      for (let i = this.level + 1; i <= lvl; i++) update[i] = this.head;
      this.level = lvl;
    }
    const newNode = new SkipNode(num, lvl);
    for (let i = 0; i <= lvl; i++) {
      newNode.next[i] = update[i].next[i];
      update[i].next[i] = newNode;
    }
  }
}```


  点击按钮查看结果
