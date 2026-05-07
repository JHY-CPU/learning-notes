# 发布订阅Pub/Sub

## 一、概念说明

Redis Pub/Sub是一种消息通信模式，发布者（Publisher）发送消息到频道（Channel），订阅者（Subscriber）接收消息。消息是即时的，不持久化。

## 二、具体用法

### 基本操作

```bash
# 订阅频道
SUBSCRIBE news:sports
# 输出: 等待消息...

# 在另一个终端发布消息
PUBLISH news:sports "比赛开始！"
# 输出: (integer) 1（收到消息的订阅者数）

# 订阅者收到消息
# 1) "message"
# 2) "news:sports"
# 3) "比赛开始！"
```

### 模式订阅

```bash
# 订阅匹配模式
PSUBSCRIBE news:*
# 接收所有news:开头频道的消息

# 发布到任意匹配频道
PUBLISH news:sports "体育新闻"
PUBLISH news:tech "科技新闻"
# 两个消息都会被收到
```

### 取消订阅

```bash
# 取消订阅
UNSUBSCRIBE news:sports

# 取消模式订阅
PUNSUBSCRIBE news:*
```

## 三、实际应用

```bash
# 实时通知
SUBSCRIBE notifications:user:1001
PUBLISH notifications:user:1001 "您有新消息"

# 实时聊天
SUBSCRIBE chat:room1
PUBLISH chat:room1 "大家好"

# 配置更新通知
SUBSCRIBE config:updates
PUBLISH config:updates "reload"
```

## 四、注意事项

1. **消息不持久化**：订阅者离线时的消息会丢失
2. **无消息确认**：发布者不知道订阅者是否收到
3. **阻塞订阅**：SUBSCRIBE会阻塞客户端
4. **性能影响**：大量订阅者会影响发布性能
5. **替代方案**：需要可靠消息使用Stream
