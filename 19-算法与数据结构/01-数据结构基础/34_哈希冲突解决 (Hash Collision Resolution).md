# Hash Collision Resolution

### 什么是哈希冲突

当两个不同的键经过哈希函数计算后得到相同的索引位置时，就发生了哈希冲突。解决冲突的策略直接影响哈希表的性能和实现复杂度。

### 关键特性

- **链地址法**：每个桶维护一个链表，冲突的键串在链表上
- **开放地址法**：冲突时向后探测空位，包括线性探测、二次探测、双重哈希
- **链地址法**实现简单，适合高负载因子；**开放地址法**缓存友好，适合低负载因子

### 时间与空间复杂度

| 方法 | 平均查找 | 最坏查找 | 空间 |
|------|---------|---------|------|
| 链地址法 | O(1+α) | O(n) | 链表指针开销 |
| 线性探测 | O(1/(1-α)) | O(n) | 无额外指针 |
| 双重哈希 | O(1/(1-α)) | O(n) | 无额外指针 |

α 为负载因子。

### 适用场景 vs 替代方案

- **链地址法**：Java HashMap 默认方案，实现简单
- **开放地址法**：Redis 字典在负载因子低时使用
- **布谷鸟哈希**：最坏情况 O(1) 查找，但插入可能失败
- **Robin Hood 哈希**：减少探测序列长度的方差

### 常见陷阱

- 线性探测产生聚集现象，性能下降快
- 开放地址法的删除需要标记（tombstone），不能直接置空
- 负载因子过高时任何方案性能都会严重退化

```
// 开放地址法（线性探测）
class OpenHash {
  constructor(n=10) { this.keys = new Array(n); this.vals = new Array(n); this.n = n; }
  _h(k) { return k.toString().split('').reduce((a,c)=>a+c.charCodeAt(0),0) % this.n; }
  set(k, v) {
    let i = this._h(k);
    while (this.keys[i] !== undefined && this.keys[i] !== k) i = (i+1) % this.n;
    this.keys[i] = k; this.vals[i] = v;
  }
  get(k) {
    let i = this._h(k);
    while (this.keys[i] !== undefined) { if (this.keys[i] === k) return this.vals[i]; i = (i+1) % this.n; }
    return undefined;
  }
}
```


### 实际应用

- **Java HashMap**：链地址法 + 红黑树优化（链表长度超过 8 时转树）
- **Redis 字典**：负载因子 > 1 时使用链地址法，< 1 时渐进式 rehash
- **Python dict**：开放地址法，使用伪随机探测序列
- **Google SwissTable**：SIMD 加速的开放地址法，性能极优

  点击按钮查看结果
