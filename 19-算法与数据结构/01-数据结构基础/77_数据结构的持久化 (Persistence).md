# Persistence

### 什么是持久化数据结构

持久化数据结构在修改时保留旧版本，每次"修改"产生新版本而不会破坏旧版本。支持回溯到历史状态，类似版本控制。

### 关键特性

- **不可变性**：修改产生新节点，旧节点保持不变
- **路径复制**：只复制从根到修改点的路径，共享未修改部分
- **空间效率**：多个版本共享大部分数据，增量存储
- **快照**：O(1) 保存当前状态的快照

### 时间与空间复杂度

| 操作 | 时间复杂度 | 空间复杂度 |
|------|-----------|-----------|
| 修改（路径复制） | O(h) | O(h) 每次 |
| 查询 | O(h) | O(1) |
| 版本切换 | O(1) | O(1) |

h 为数据结构深度（树高）。

### 适用场景 vs 替代方案

- **撤销操作**：编辑器的多级撤销功能
- **版本历史**：Git 式的数据版本管理
- **函数式编程**：不可变数据结构是函数式语言的基础
- **替代**：深拷贝简单但 O(n) 空间，不适合频繁快照

### 常见陷阱

- 实现时忘记共享未修改子树，退化为深拷贝
- 版本过多时内存增长，需要垃圾回收旧版本
- JavaScript 中实现不可变性需要 Object.freeze 或专用库

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
}
```


### 实际应用

- **Git**：每次提交创建新版本，共享未修改的文件
- **Immutable.js**：Facebook 的 JavaScript 持久化数据结构库
- **Clojure**：语言层面内置持久化向量和哈希映射
- **数据库 MVCC**：PostgreSQL 用多版本并发控制保留历史版本

  点击按钮查看结果
