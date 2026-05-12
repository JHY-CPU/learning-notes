# Circular Queue

### 什么是循环队列

循环队列使用固定大小的数组，通过取模运算将数组首尾相连，用 head 和 tail 两个指针管理入队出队。避免了普通数组队列中 dequeue 时的数据搬移，所有操作均为 O(1)。

### 关键特性

- **环形缓冲**：tail 到达数组末尾后回到开头，充分利用空间
- **牺牲一个位置**：通常保留一个空位区分队满和队空
- **固定容量**：创建时确定大小，不可动态扩容

### 时间与空间复杂度

| 操作 | 时间复杂度 | 空间复杂度 |
|------|-----------|-----------|
| enqueue | O(1) | O(k) 预分配 |
| dequeue | O(1) | - |
| front | O(1) | - |
| isFull | O(1) | - |

### 适用场景 vs 替代方案

- **适用**：固定缓冲区场景，如音视频流缓冲、IO 缓冲
- **替代**：需要动态扩容时用链表队列或 ArrayDeque
- **替代**：需要优先级时用优先队列（堆实现）

### 常见陷阱

- 区分队满和队空的条件：(tail + 1) % cap === head 表示满
- head === tail 表示空，但这也意味着要浪费一个存储位置
- 取模运算忘记处理负数索引的情况

```
class CircularQueue {
  constructor(k) {
    this.data = new Array(k);
    this.head = 0;
    this.tail = 0;
    this.cap = k;
  }
  enqueue(x) {
    if ((this.tail + 1) % this.cap === this.head) return false;
    this.data[this.tail] = x;
    this.tail = (this.tail + 1) % this.cap;
    return true;
  }
  dequeue() {
    if (this.head === this.tail) return false;
    this.head = (this.head + 1) % this.cap;
    return true;
  }
  front() { return this.head === this.tail ? -1 : this.data[this.head]; }
  isFull() { return (this.tail + 1) % this.cap === this.head; }
}
```


### 实际应用

- **音频播放器**：环形缓冲区平滑音频数据流
- **CPU 调度**：操作系统用循环队列实现时间片轮转调度
- **网络通信**：TCP 滑动窗口本质上是循环缓冲区的变体
- **嵌入式系统**：固定内存环境下高效管理数据流

  点击按钮查看结果
