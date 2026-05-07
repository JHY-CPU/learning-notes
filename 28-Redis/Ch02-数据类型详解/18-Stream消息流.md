# Stream消息流

## 一、概念说明

Stream是Redis 5.0引入的数据类型，专门用于消息队列场景。它支持消息持久化、消费者组、ACK确认机制，比List实现的消息队列更强大可靠。

## 二、具体用法

### 添加消息

```bash
# 添加消息（自动生成ID）
XADD mystream * name "张三" action "login" time "2024-01-01"
# 输出: "1704067200000-0"（消息ID）

# 添加多条
XADD mystream * name "李四" action "purchase" amount 99.9
XADD mystream * name "王五" action "logout"

# 指定最大长度
XADD mystream MAXLEN 1000 * field value

# 近似裁剪（更高效）
XADD mystiream MAXLEN ~ 1000 * field value
```

### 读取消息

```bash
# 读取所有消息
XRANGE mystream - +
# 输出: 所有消息

# 读取最新消息
XREVRANGE mystream + - COUNT 5
# 输出: 最新的5条消息

# 读取指定范围
XRANGE mystream 1704067200000-0 1704067200000-5

# 从某ID开始读取新消息
XREAD COUNT 10 STREAMS mystream 0
# 输出: 从ID 0开始的10条消息

# 阻塞式读取（等待新消息）
XREAD BLOCK 5000 COUNT 1 STREAMS mystream $
# $表示最新的ID
# 阻塞5秒等待新消息
```

### 消息管理

```bash
# 获取流信息
XINFO STREAM mystream
# 输出: 长度、消费者组等信息

# 获取消息数量
XLEN mystream
# 输出: (integer) 3

# 删除消息
XDEL mystream 1704067200000-0
# 输出: (integer) 1

# 裁剪流
XTRIM mystream MAXLEN 100
# 输出: 裁剪后保留的元素数

# 获取消息详情
XINFO STREAM mystream FULL
```

## 三、Stream vs List消息队列

| 特性 | Stream | List |
|------|--------|------|
| 消息持久化 | 支持 | 支持 |
| 消费者组 | 支持 | 不支持 |
| ACK确认 | 支持 | 不支持 |
| 消息回溯 | 支持 | 不支持 |
| 消息ID | 自动生成时间戳 | 无 |
| 广播消费 | 支持 | 不支持 |

## 四、实际应用

```bash
# 事件日志
XADD events:log * type "error" service "api" message "Connection timeout"

# 订单流
XADD orders:pending * order_id 1001 user_id 5001 amount 99.9

# 阻塞消费
XREAD BLOCK 0 STREAMS orders:pending $
# 阻塞等待新订单
```

## 五、注意事项与常见陷阱

1. **消息ID是时间戳**：毫秒级时间戳+序列号
2. **MAXLEN裁剪**：限制内存使用，避免流无限增长
3. **阻塞读取不阻塞Redis**：使用IO多路复用
4. **消息不可修改**：只能添加和删除
5. **消费者组需要ACK**：不ACK的消息会被重新投递
6. **适合日志场景**：顺序写入、消费者组消费
