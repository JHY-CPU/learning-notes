# RabbitMQ 概述

## 一、什么是 RabbitMQ

RabbitMQ 是一个开源的**消息代理（Message Broker）**，实现了 AMQP（Advanced Message Queuing Protocol）协议。它使用 Erlang 语言开发，以高可靠性、灵活路由和丰富的插件生态著称。

```java
// RabbitMQ 的核心模型
Producer -> Exchange -> Binding -> Queue -> Consumer
```

## 二、AMQP 协议

AMQP 是一个开放标准的应用层协议，定义了消息在客户端和代理之间的交互方式。

```
AMQP 协议层次:
┌─────────────────────────┐
│   AMQP Model            │  消息路由、队列、确认
├─────────────────────────┤
│   AMQP Framing          │  帧格式、信道复用
├─────────────────────────┤
│   AMQP Protocol         │  连接管理、认证
├─────────────────────────┤
│   TCP/IP                │  网络传输
└─────────────────────────┘
```

```java
// AMQP 核心概念对应 RabbitMQ 实现
// Connection - TCP 连接
Connection connection = factory.newConnection();

// Channel - 轻量级虚拟连接（复用 TCP 连接）
Channel channel = connection.createChannel();

// Exchange - 消息路由
channel.exchangeDeclare("my-exchange", "direct");

// Queue - 消息存储
channel.queueDeclare("my-queue", true, false, false, null);

// Binding - 绑定关系
channel.queueBind("my-queue", "my-exchange", "routing-key");
```

## 三、Erlang 语言特点

RabbitMQ 选择 Erlang 有以下原因：

```erlang
%% Erlang 的特点
%% 1. 天然支持并发 - 轻量级进程模型
%% 2. 高可用 - "Let it crash" 哲学，进程隔离
%% 3. 热代码升级 - 无需停机更新
%% 4. 分布式 - 内置分布式通信支持

%% 一个简单的 Erlang 进程示例
-module(hello).
-export([start/0]).

start() ->
    Pid = spawn(fun() -> loop() end),
    Pid ! {self(), "Hello RabbitMQ"}.

loop() ->
    receive
        {From, Message} ->
            io:format("Received: ~s~n", [Message]),
            loop()
    end.
```

## 四、RabbitMQ 核心特性

| 特性 | 说明 |
|------|------|
| 可靠性 | 持久化、Publisher Confirm、Consumer ACK |
| 灵活路由 | 4 种 Exchange 类型，支持复杂路由规则 |
| 集群 | 多节点集群，镜像队列 / Quorum Queue |
| 高可用 | 镜像复制、队列镜像 |
| 多语言客户端 | Java、Python、Go、Node.js、C# 等 |
| 管理界面 | Web UI + HTTP API |
| 插件系统 | 延迟消息、Shovel、Federation 等 |

## 五、版本演进

```yaml
版本历史:
  3.x: 经典版本，镜像队列
  3.8+: 引入 Quorum Queue (Raft 协议)
  3.12+: Stream 队列 (类似 Kafka 日志)
  3.13+: 默认 Quorum Queue，性能提升
  4.0: 新一代架构，Khepri 元数据存储
```

## 六、注意事项

1. **Erlang 运维对 Java 团队是额外学习成本**，考虑使用 Docker 部署
2. **单队列消费能力有限**，需要通过多队列并行提升吞吐
3. **内存管理由 Erlang VM 控制**，需要合理设置内存阈值
4. **网络分区处理策略要慎重选择**，`pause_minority` 是最安全的选项
5. **生产环境推荐使用 Quorum Queue**，取代传统的镜像队列
