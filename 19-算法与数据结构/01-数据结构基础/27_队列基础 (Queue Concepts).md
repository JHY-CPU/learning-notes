# Queue Concepts

### 什么是队列

队列（Queue）是一种先进先出（FIFO, First In First Out）的线性数据结构。新元素在队尾（rear）加入，从队头（front）移除，如同排队等候服务。

### 关键特性

- **FIFO 顺序**：最先入队的元素最先出队
- **两端操作**：队尾入队（enqueue），队头出队（dequeue）
- **公平性**：保证元素按到达顺序被处理

### 时间与空间复杂度

| 操作 | 时间复杂度 | 说明 |
|------|-----------|------|
| enqueue | O(1) | 队尾添加元素（用数组 push） |
| dequeue | O(1) 链表/O(n) 数组 | 队头移除元素 |
| front/peek | O(1) | 查看队头元素 |
| isEmpty | O(1) | 判断是否为空 |

注意：用普通数组实现时，dequeue（shift）是 O(n) 操作，应改用链表或循环队列。

### 适用场景 vs 替代方案

- **广度优先搜索**：队列是 BFS 的核心，栈用于 DFS
- **任务调度**：按请求到达顺序处理，如消息队列
- **缓冲区**：生产者-消费者模型中的数据缓冲
- **替代**：需要双端操作时用双端队列（Deque），需要优先级时用堆

### 常见陷阱

- 用数组 shift() 实现出队导致 O(n) 性能问题
- 未判空就执行 dequeue，导致 undefined 或异常
- 混淆队列和栈的使用场景

```
class Queue {
  constructor() { this.items = []; }
  enqueue(x) { this.items.push(x); }
  dequeue() { return this.items.shift(); }
  front() { return this.items[0]; }
  isEmpty() { return this.items.length === 0; }
  size() { return this.items.length; }
}
const q = new Queue();
q.enqueue(1); q.enqueue(2); q.enqueue(3);
console.log(q.dequeue()); // 1
console.log(q.front());   // 2
```


### 实际应用

- **打印队列**：多任务提交时按顺序处理打印请求
- **消息中间件**：RabbitMQ、Kafka 使用队列管理消息传递
- **Web 服务器**：请求队列管理并发连接的处理顺序
- **BFS 算法**：图的最短路径、层序遍历等

  点击按钮查看结果
