# Hash Table Concepts

### 什么是哈希表

哈希表（Hash Table）通过哈希函数将键（key）映射到数组的存储位置（桶），实现近乎 O(1) 的查找、插入和删除。是计算机科学中最重要的数据结构之一。

### 关键特性

- **键值对存储**：每个键唯一对应一个值
- **哈希函数**：将任意大小的键转换为固定范围的数组索引
- **负载因子**：已存储元素数/桶数，超过阈值需要扩容
- **无序性**：元素存储顺序与插入顺序无关（JS Map 除外）

### 时间与空间复杂度

| 操作 | 平均 | 最坏 | 说明 |
|------|------|------|------|
| 插入 | O(1) | O(n) | 最坏时所有键冲突到同一桶 |
| 查找 | O(1) | O(n) | 取决于冲突链长度 |
| 删除 | O(1) | O(n) | 链地址法删除较简单 |
| 空间 | O(n) | O(n) | 需要额外桶空间 |

### 适用场景 vs 替代方案

- **快速查找**：哈希表 vs 二叉搜索树（O(1) vs O(log n)）
- **需要有序遍历**：用 TreeMap/TreeSet（红黑树）
- **键是整数且范围小**：直接用数组更高效
- **只需要判断存在**：用 HashSet 或布隆过滤器

### 常见陷阱

- 哈希函数设计不当导致大量冲突，性能退化
- 对象作为键时未重写 hashCode/equals，导致查找失败
- 并发环境下未加锁可能导致数据损坏

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
}
```


### 实际应用

- **数据库索引**：B+树和哈希索引加速查询
- **缓存系统**：Redis、Memcached 底层大量使用哈希表
- **编译器符号表**：快速查找变量名对应的类型和地址
- **关联数组**：Python dict、JavaScript Object/Map、Java HashMap

  点击按钮查看结果
