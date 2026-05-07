# Exchange 类型

## 一、Exchange 概述

Exchange（交换机）是 RabbitMQ 的消息路由组件，负责根据路由规则将消息分发到绑定的队列。生产者不直接发送消息到队列，而是发送到 Exchange。

```
Producer -> Exchange -> (Binding Rules) -> Queue1
                                           Queue2
                                           Queue3
```

## 二、Direct Exchange（直连交换机）

精确匹配 Routing Key，消息路由到 Routing Key 完全匹配的队列。

```java
// Direct Exchange 配置
@Configuration
public class DirectExchangeConfig {

    @Bean
    public DirectExchange orderExchange() {
        return new DirectExchange("order-direct-exchange", true, false);
    }

    @Bean
    public Queue paidQueue() {
        return QueueBuilder.durable("order-paid-queue").build();
    }

    @Bean
    public Queue shippedQueue() {
        return QueueBuilder.durable("order-shipped-queue").build();
    }

    @Bean
    public Binding paidBinding() {
        return BindingBuilder.bind(paidQueue())
            .to(orderExchange())
            .with("order.paid");  // 精确匹配
    }

    @Bean
    public Binding shippedBinding() {
        return BindingBuilder.bind(shippedQueue())
            .to(orderExchange())
            .with("order.shipped");
    }
}

// 发送消息
rabbitTemplate.convertAndSend("order-direct-exchange", "order.paid", order);
// -> 只有 order-paid-queue 收到消息
```

## 三、Topic Exchange（主题交换机）

使用通配符匹配 Routing Key，支持 `*`（匹配一个单词）和 `#`（匹配零或多个单词）。

```java
// Topic Exchange 配置
@Configuration
public class TopicExchangeConfig {

    @Bean
    public TopicExchange eventExchange() {
        return new TopicExchange("event-topic-exchange", true, false);
    }

    @Bean
    public Queue allOrdersQueue() {
        return QueueBuilder.durable("all-orders-queue").build();
    }

    @Bean
    public Queue paymentQueue() {
        return QueueBuilder.durable("payment-events-queue").build();
    }

    // 匹配所有订单事件
    @Bean
    public Binding allOrdersBinding() {
        return BindingBuilder.bind(allOrdersQueue())
            .to(eventExchange())
            .with("order.#");  // order.created, order.paid, order.shipped 都匹配
    }

    // 匹配所有支付相关事件
    @Bean
    public Binding paymentBinding() {
        return BindingBuilder.bind(paymentQueue())
            .to(eventExchange())
            .with("#.payment.*");  // order.payment.success, refund.payment.failed
    }
}

// 通配符说明
// order.*      匹配 order.created, order.paid（不匹配 order.payment.success）
// order.#      匹配 order.created, order.payment.success（任意层级）
// *.payment.*  匹配 order.payment.success, refund.payment.failed
```

## 四、Fanout Exchange（扇出交换机）

忽略 Routing Key，将消息广播到所有绑定的队列。

```java
// Fanout Exchange - 广播模式
@Configuration
public class FanoutExchangeConfig {

    @Bean
    public FanoutExchange broadcastExchange() {
        return new FanoutExchange("broadcast-fanout-exchange", true, false);
    }

    @Bean
    public Queue logQueue() {
        return QueueBuilder.durable("log-queue").build();
    }

    @Bean
    public Queue monitorQueue() {
        return QueueBuilder.durable("monitor-queue").build();
    }

    @Bean
    public Queue archiveQueue() {
        return QueueBuilder.durable("archive-queue").build();
    }

    // 所有队列都绑定到 Fanout Exchange，无需 routing-key
    @Bean
    public Binding logBinding() {
        return BindingBuilder.bind(logQueue()).to(broadcastExchange());
    }

    @Bean
    public Binding monitorBinding() {
        return BindingBuilder.bind(monitorQueue()).to(broadcastExchange());
    }

    @Bean
    public Binding archiveBinding() {
        return BindingBuilder.bind(archiveQueue()).to(broadcastExchange());
    }
}

// 发送消息 - 所有绑定队列都会收到
rabbitTemplate.convertAndSend("broadcast-fanout-exchange", "", "系统通知");
```

## 五、Headers Exchange（头交换机）

根据消息头属性匹配，而非 Routing Key。支持 `x-match=all`（所有头匹配）或 `x-match=any`（任一头匹配）。

```java
// Headers Exchange
@Configuration
public class HeadersExchangeConfig {

    @Bean
    public HeadersExchange headersExchange() {
        return new HeadersExchange("headers-exchange", true, false);
    }

    @Bean
    public Queue vipQueue() {
        return QueueBuilder.durable("vip-order-queue").build();
    }

    @Bean
    public Binding vipBinding() {
        Map<String, Object> headers = new HashMap<>();
        headers.put("userType", "VIP");
        headers.put("orderType", "EXPRESS");
        return BindingBuilder.bind(vipQueue())
            .to(headersExchange())
            .whereAll(headers)  // 所有头都匹配（x-match=all）
            .match();
    }
}

// 发送带头信息的消息
MessageProperties props = new MessageProperties();
props.setHeader("userType", "VIP");
props.setHeader("orderType", "EXPRESS");
Message message = new Message(JSON.toJSONString(order).getBytes(), props);
rabbitTemplate.send("headers-exchange", "", message);
```

## 六、Exchange 类型选择

| 类型 | 路由方式 | 适用场景 |
|------|---------|---------|
| Direct | 精确匹配 routing-key | 点对点消息、任务分发 |
| Topic | 通配符匹配 | 日志路由、事件分类 |
| Fanout | 广播 | 系统通知、缓存刷新 |
| Headers | 消息头匹配 | 复杂路由条件（很少使用） |

## 七、注意事项

1. **Fanout 性能最高**，因为不需要路由匹配计算
2. **Topic 的通配符匹配有一定性能开销**，大量 binding 时注意性能
3. **Headers Exchange 性能最差**，生产环境很少使用
4. **默认 Exchange（空字符串）是 Direct 类型**，routing-key 等于队列名
5. **Exchange 声明是幂等的**，重复声明相同属性的 Exchange 不会报错
