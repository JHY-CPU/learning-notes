# List阻塞操作

## 一、概念说明

Redis的List支持阻塞式弹出操作（BLPOP/BRPOP），在列表为空时客户端会阻塞等待，直到有元素可用或超时。这是实现简单消息队列的基础。

## 二、具体用法

### BLPOP与BRPOP

```bash
# 阻塞式左弹出（超时时间单位：秒）
BLPOP mylist 0
# 阻塞直到有元素
# 输出: 1) "mylist" 2) "element"

# 设置超时时间
BLPOP mylist 5
# 最多阻塞5秒
# 输出: (nil)（超时无数据）

# 阻塞式右弹出
BRPOP mylist 0
# 阻塞直到有元素可用

# 从多个列表阻塞弹出
BLPOP list1 list2 list3 0
# 按顺序检查，第一个有数据的列表弹出
```

### 消息队列实现

```bash
# 生产者
LPUSH queue:orders '{"id":1001,"product":"iPhone","amount":9999}'
LPUSH queue:orders '{"id":1002,"product":"iPad","amount":5999}'

# 消费者（阻塞等待）
BRPOP queue:orders 0
# 输出: 1) "queue:orders"
#       2) "{\"id\":1001,\"product\":\"iPhone\",\"amount\":9999}"

# 多消费者模式
# 每个消费者监听同一个队列
# BRPOP保证每个消息只被一个消费者处理
```

### BRPOPLPUSH/BLMOVE

```bash
# 原子性的弹出并推入另一个列表
BRPOPLPUSH source destination 0
# 从source弹出尾部元素，推入destination头部

# 应用：可靠消息队列
# 消费时先移动到处理中列表
BRPOPLPUSH queue:pending queue:processing 0

# 处理完成后从处理中列表删除
LREM queue:processing 1 "message"

# 处理失败时重新放回pending
RPOPLPUSH queue:processing queue:pending
```

### 阻塞机制原理

```bash
# 阻塞原理
# 1. 客户端发送BLPOP命令
# 2. 列表为空时，客户端被放入等待队列
# 3. 有新元素时，通知等待的客户端
# 4. 超时后返回nil

# 注意：Redis是单线程的
# 阻塞操作不会阻塞其他客户端
# 使用IO多路复用管理多个阻塞连接
```

## 三、应用场景

```bash
# 1. 简单消息队列
LPUSH task_queue "task1"
BRPOP task_queue 0

# 2. 工作队列
LPUSH work_queue "job1"
# 多个worker竞争消费
BRPOP work_queue 0

# 3. 事件通知
LPUSH events "user_login"
BRPOP events 0

# 4. 延迟处理
# 先存入延迟队列，定时任务消费
BRPOP delayed_queue 0
```

## 四、注意事项与常见陷阱

1. **超时时间0表示永久阻塞**：直到有数据可用
2. **多个列表按顺序检查**：第一个有数据的列表优先
3. **阻塞不消耗CPU**：使用epoll等IO多路复用
4. **连接断开时自动取消阻塞**：客户端重连后重新发送
5. **BRPOPLPUSH是原子操作**：保证弹出和推入的原子性
6. **消息丢失问题**：简单队列没有ACK机制，处理失败消息丢失
