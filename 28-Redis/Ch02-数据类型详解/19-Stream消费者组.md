# Stream消费者组

## 一、概念说明

Stream的消费者组（Consumer Group）允许多个消费者协同消费同一个Stream，每个消息只会被一个消费者处理。支持消息确认（ACK）、未确认消息回溯等功能，是可靠消息队列的核心。

## 二、具体用法

### 创建消费者组

```bash
# 添加测试数据
XADD orders * order_id 1001 amount 99.9
XADD orders * order_id 1002 amount 199.9
XADD orders * order_id 1003 amount 299.9

# 创建消费者组
XGROUP CREATE orders mygroup 0
# 输出: OK

# 从最新消息开始创建
XGROUP CREATE orders mygroup2 $
# 输出: OK

# 创建消费者
XGROUP CREATECONSUMER orders mygroup consumer1
XGROUP CREATECONSUMER orders mygroup consumer2
```

### 消费消息

```bash
# 消费者组消费
XREADGROUP GROUP mygroup consumer1 COUNT 2 STREAMS orders >
# > 表示获取未投递的消息
# 输出: 2条消息

# 再次消费（获取剩余消息）
XREADGROUP GROUP mygroup consumer2 COUNT 2 STREAMS orders >

# 阻塞式消费
XREADGROUP GROUP mygroup consumer1 BLOCK 5000 COUNT 1 STREAMS orders >
```

### 消息确认

```bash
# 确认消息处理完成
XACK orders mygroup 1704067200000-0
# 输出: (integer) 1

# 批量确认
XACK orders mygroup 1704067200000-0 1704067200000-1
# 输出: (integer) 2

# 查看未确认消息
XPENDING orders mygroup
# 输出: 消费者组待确认消息概览

# 查看详细未确认消息
XPENDING orders mygroup - + 10 consumer1
# 输出: 指定消费者的未确认消息列表
```

### 处理未确认消息

```bash
# 获取未确认消息
XPENDING orders mygroup - + 10
# 输出: 未确认消息列表

# 认领未确认消息（消费者挂了后重新消费）
XCLAIM orders mygroup consumer2 0 1704067200000-0
# 将消息分配给consumer2

# 自动认领（超过指定时间未确认）
XAUTOCLAIM orders mygroup consumer2 60000 0-0 COUNT 10
# Redis 6.2+
```

### 消费者组管理

```bash
# 查看消费者组信息
XINFO GROUPS orders
# 输出: 所有消费者组的详细信息

# 查看消费者信息
XINFO CONSUMERS orders mygroup
# 输出: 组内所有消费者的信息

# 删除消费者
XGROUP DELCONSUMER orders mygroup consumer1

# 删除消费者组
XGROUP DESTROY orders mygroup

# 设置消费者组待处理消息
XGROUP SETID orders mygroup 0
```

## 三、消费者组工作原理

```
消息流: [msg1] [msg2] [msg3] [msg4] [msg5]
         |      |      |
         v      v      v
消费者组 mygroup
├── consumer1: 消费 msg1, msg4 (已ACK msg1)
├── consumer2: 消费 msg2, msg5 (未ACK)
└── consumer3: 消费 msg3 (已ACK)

待处理: msg2, msg4, msg5
```

## 四、注意事项与常见陷阱

1. **必须ACK**：不ACK的消息会重新投递
2. **消费者组不能重复创建**：使用MKSTREAM选项自动创建流
3. **阻塞消费**：适合实时处理场景
4. **消息丢失风险**：消费者处理后、ACK前崩溃会导致重复消费
5. **XAUTOCLAIM自动恢复**：自动认领超时未确认的消息
6. **性能考虑**：大量未确认消息会影响性能
