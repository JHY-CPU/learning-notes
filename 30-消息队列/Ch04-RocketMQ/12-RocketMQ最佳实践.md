# RocketMQ 最佳实践

## 一、生产者最佳实践

```java
// 1. Producer 实例复用，不要频繁创建
// 2. 消息体不要超过 4MB
// 3. 合理设置超时和重试
producer.setSendMsgTimeout(3000);
producer.setRetryTimesWhenSendFailed(2);

// 4. 重要消息使用同步发送 + 同步刷盘
producer.send(msg);  // 同步

// 5. 日志类消息使用单向发送
producer.sendOneway(msg);
```

## 二、消费者最佳实践

```java
// 1. 消费者要做好幂等
@RocketMQMessageListener(topic = "OrderTopic")
public class Consumer implements RocketMQListener<Order> {
    @Override
    public void onMessage(Order order) {
        if (!idempotentService.tryProcess(order.getId())) return;
        processOrder(order);
    }
}

// 2. 合理设置消费线程数
@RocketMQMessageListener(
    consumeThreadNumber = 20,
    maxReconsumeTimes = 16
)

// 3. 消费失败不要吞掉异常
public void onMessage(Order order) {
    try {
        process(order);
    } catch (BusinessException e) {
        // 业务异常 - 正常确认
        log.warn("业务异常: {}", e.getMessage());
    } catch (Exception e) {
        // 系统异常 - 触发重试
        throw e;
    }
}
```

## 三、Topic 设计规范

```yaml
命名规范:
  格式: {业务}-{类型}
  示例: order-topic, payment-callback, log-topic

队列数:
  默认: 8
  高吞吐: 16-32
  顺序消息: 根据并行度设置

Tag 设计:
  格式: {业务细分}
  示例: VIP, NORMAL, EXPRESS
  限制: 单个订阅不超过 64 个 Tag
```

## 四、可靠性保障

```java
// 生产端
// 1. 同步发送 + 等待确认
SendResult result = producer.send(msg);
if (result.getSendStatus() != SendStatus.SEND_OK) {
    // 记录失败消息，后续重试
}

// 2. 事务消息
rocketMQTemplate.sendMessageInTransaction("Topic", msg, arg);

// 消费端
// 1. 手动控制消费成功/失败
public Action consume(Message message) {
    try {
        process(message);
        return Action.CommitMessage;
    } catch (Exception e) {
        return Action.ReconsumeLater;
    }
}
```

## 五、监控与运维

```bash
# 关键监控指标
# 1. 消费延迟
mqadmin consumerProgress -n localhost:9876 -g OrderGroup

# 2. Topic 状态
mqadmin topicStatus -n localhost:9876 -t OrderTopic

# 3. Broker 状态
mqadmin brokerStatus -n localhost:9876 -b 192.168.1.10:10911

# 4. 消息轨迹
mqadmin queryMsgById -n localhost:9876 -i MSG_ID
```

## 六、常见问题

```yaml
消息堆积:
  原因: 消费能力不足
  解决: 增加消费者实例、提升消费线程数、批量消费

消息丢失:
  原因: 异步刷盘、异步复制
  解决: SYNC_MASTER + SYNC_FLUSH

消息重复:
  原因: 重试机制、网络抖动
  解决: 消费端幂等处理

事务消息超时:
  原因: 回查未返回确定状态
  解决: 确保 checkLocalTransaction 能判断状态
```

## 七、注意事项

1. **Producer 和 Consumer 的 Group 命名要规范**
2. **不要在消息体中存储大对象**
3. **事务消息做好回查接口**
4. **监控消费延迟是最有效的预警手段**
5. **5.0 版本升级要做好兼容性测试**
