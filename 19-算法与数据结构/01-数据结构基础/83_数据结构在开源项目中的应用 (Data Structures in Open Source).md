# 84-数据结构在开源项目中的应用 (Data Structures in Open Source)

开源项目中大量使用高级数据结构来优化性能和内存使用。

## V8 引擎中的数据结构

```javascript
// V8 使用隐藏类（Hidden Class）优化对象属性访问
// 类似于固定偏移量的数组访问，而非哈希表查找

// V8 字符串表示：
// - Short String (< 12 字符): 直接存储在对象内
// - Long String: 使用 ConsString（树状拼接）或 ExternalString
// - Sliced String: 子字符串不复制，引用原字符串 + 偏移量

// V8 数组优化：
// - PACKED 元素: 连续存储，O(1) 访问
// - HOLEY 元素: 有空洞，需要查原型链
// - 快数组 vs 慢数组：小数组用连续存储，大/稀疏用哈希表
```

## Redis 中的数据结构

```javascript
// Redis 有序集合（zset）使用跳表 + 哈希表
// - 跳表: O(log N) 范围查询和排名
// - 哈希表: O(1) 成员查找

// Redis 列表：
// - 元素少时: 压缩列表（ziplist）连续内存存储
// - 元素多时: 双向链表
// - Redis 7.0+: listpack 替代 ziplist

// Redis 哈希：
// - 字段少时: 压缩列表
// - 字段多时: 哈希表
```

## React Fiber 中的链表

```javascript
// React Fiber 架构使用链表实现可中断的渲染
class FiberNode {
  constructor(tag, key) {
    this.tag = tag;          // 组件类型标记
    this.key = key;          // 唯一标识
    this.type = null;        // 函数/类/原生DOM
    this.child = null;       // 第一个子 Fiber
    this.sibling = null;     // 兄弟 Fiber
    this.return = null;      // 父 Fiber
    this.stateNode = null;   // 对应的 DOM 节点或组件实例
    this.memoizedState = null;
    this.pendingProps = null;
    this.memoizedProps = null;
    this.effectTag = null;   // 副作用标记
  }
}

// Fiber 树遍历（非递归，可中断）
function workLoop(fiber) {
  while (fiber) {
    processFiber(fiber);
    // 优先找子节点
    if (fiber.child) { fiber = fiber.child; continue; }
    // 没有子节点找兄弟
    while (fiber) {
      if (fiber.sibling) { fiber = fiber.sibling; break; }
      fiber = fiber.return; // 回溯
    }
  }
}
```

## Linux 内核中的数据结构

```cpp
// Linux 使用红黑树管理进程调度
// - CFS（完全公平调度器）用红黑树按虚拟运行时间排序
// - O(log n) 找到最需要运行的进程

// Linux 页表：多级页表（类似多叉树）
// - 4 级页表（x86_64）: PML4 -> PDPT -> PD -> PT
// - 减少内存占用（稀疏地址空间只分配实际使用的页表）

// Linux 内存分配器（SLAB）：
// - 每种对象类型一个缓存池
// - 空闲对象用链表管理
```

## Google Protobuf

```cpp
// Protocol Buffers 使用变长整数编码（Varint）
// - 小整数用更少字节
// - 每字节最高位表示是否继续

// 字段编码：[字段编号 << 3 | 类型] + 值
// 使用哈希表（FieldDescriptor）加速字段查找
```

## HashMap 在各语言中的实现

```
Java HashMap:
  - 数组 + 链表 + 红黑树
  - 链表长度 > 8 且数组长度 >= 64 时转红黑树
  - 容量始终为 2 的幂次

Python dict:
  - 开放地址法（二次探测）
  - 紧凑字典（Python 3.6+）保持插入顺序
  - 共享键（多实例共享 key 数组）

Go map:
  - 桶数组 + 溢出桶链表
  - 每个桶存 8 个键值对
  - 渐进式扩容

Rust HashMap:
  - 基于 SwissTable（Google Abseil）
  - SIMD 优化批量探测
  - 开放地址法 + 控制字节
```

## 总结

开源项目中数据结构选择的关键考量：

1. **性能特征**：根据访问模式选择（随机 vs 顺序，读多 vs 写多）
2. **内存布局**：缓存友好性至关重要（连续 > 链式）
3. **并发需求**：跳表优于红黑树（CAS 友好）
4. **渐进式处理**：大操作拆分小步骤（渐进式 rehash）
5. **自适应切换**：小数据量用简单结构，大数据量用复杂结构
