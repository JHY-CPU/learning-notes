# Concurrent Safety

### 什么是并发安全

并发环境下多个线程同时操作数据结构可能导致数据竞争、丢失更新、死锁等问题。并发安全的数据结构保证在多线程环境下操作的正确性。

### 关键特性

- **互斥锁**：同一时刻只有一个线程能访问数据
- **读写锁**：允许多个读操作并行，写操作独占
- **CAS（Compare-And-Swap）**：无锁并发的基础原语
- **无锁数据结构**：利用 CAS 实现无需锁的线程安全结构

### 并发方案对比

| 方案 | 性能 | 实现难度 | 适用场景 |
|------|------|---------|---------|
| 互斥锁 | 低 | 简单 | 通用 |
| 读写锁 | 中 | 中等 | 读多写少 |
| CAS 自旋 | 高 | 较难 | 低竞争 |
| 无锁结构 | 最高 | 很难 | 高性能需求 |

### 适用场景 vs 替代方案

- **高并发读写**：ConcurrentHashMap（分段锁）
- **高并发队列**：无锁队列（如 Disruptor）
- **低竞争**：CAS + 重试足够
- **JavaScript 单线程**：通常不需要考虑并发安全

### 常见陷阱

- 锁粒度太粗导致并发度低
- 死锁：多个锁的获取顺序不一致
- CAS 的 ABA 问题：值变了又变回原值
- 忘记释放锁导致其他线程永久等待

```
// 使用锁的线程安全队列（概念示例）
class ConcurrentQueue {
  constructor() { this.items = []; this.locked = false; }
  async enqueue(x) {
    while (this.locked) await new Promise(r => setTimeout(r, 1));
    this.locked = true;
    this.items.push(x);
    this.locked = false;
  }
  async dequeue() {
    while (this.locked) await new Promise(r => setTimeout(r, 1));
    this.locked = true;
    const val = this.items.shift();
    this.locked = false;
    return val;
  }
}
console.log('并发安全队列（使用自旋锁概念）');
```


### 实际应用

- **Java ConcurrentHashMap**：分段锁实现高并发哈希表
- **Go sync.Map**：读多写少场景的优化并发 Map
- **Disruptor**：LMAX 的无锁高性能环形队列
- **Linux 内核**：RCU（Read-Copy-Update）实现无锁读取

  点击按钮查看结果
