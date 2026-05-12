# Object Pool

### 什么是对象池

对象池模式预先创建一组对象放入池中，需要时取出（acquire），用完归还（release），避免频繁的内存分配和垃圾回收。适合创建代价高或频繁创建销毁的场景。

### 关键特性

- **复用**：避免重复创建销毁的性能开销
- **固定容量**：池有最大大小，超出后需等待或创建新对象
- **重置状态**：归还时重置对象状态，保证下次使用干净
- **工厂模式**：通过 factory 函数创建新对象

### 时间与空间复杂度

| 操作 | 时间复杂度 | 说明 |
|------|-----------|------|
| acquire | O(1) | 从池顶取出 |
| release | O(1) | 重置后放回池中 |
| 空间 | O(cap) | 预分配 cap 个对象 |

### 适用场景 vs 替代方案

- **数据库连接池**：连接创建代价高，必须复用
- **游戏开发**：子弹、粒子等频繁创建销毁的对象
- **网络编程**：缓冲区复用减少 GC 压力
- **替代**：短生命周期小对象通常不需要池化

### 常见陷阱

- 归还时未重置对象状态，导致数据污染
- 池大小设置不当：太小导致频繁创建，太大浪费内存
- 并发环境下需要线程安全的获取和归还

```
class ObjectPool {
  constructor(factory, reset, initialSize=10) {
    this.factory = factory;
    this.reset = reset;
    this.pool = [];
    for (let i = 0; i < initialSize; i++) this.pool.push(factory());
  }
  acquire() {
    if (this.pool.length > 0) return this.pool.pop();
    return this.factory();
  }
  release(obj) {
    this.reset(obj);
    this.pool.push(obj);
  }
  size() { return this.pool.length; }
}
// 用法示例
const vecPool = new ObjectPool(
  () => ({x:0, y:0}),
  (v) => { v.x=0; v.y=0; }
);
```


### 实际应用

- **HikariCP**：Java 高性能数据库连接池
- **线程池**：ExecutorService 复用线程避免创建开销
- **游戏引擎**：Unity 对象池管理 GameObject 的创建和回收
- **HTTP 连接池**：axios/fetch 底层复用 TCP 连接

  点击按钮查看结果
