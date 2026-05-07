# Node.js amqplib 客户端

## 一、安装与连接

```bash
npm install amqplib
```

```javascript
// connection.js - 连接管理
const amqp = require('amqplib');

class RabbitMQClient {
    constructor(url = 'amqp://admin:admin123@localhost:5672') {
        this.url = url;
        this.connection = null;
        this.channel = null;
    }

    async connect() {
        this.connection = await amqp.connect(this.url);
        this.channel = await this.connection.createChannel();

        // 连接异常处理
        this.connection.on('error', (err) => {
            console.error('连接错误:', err);
            setTimeout(() => this.connect(), 5000);
        });

        this.connection.on('close', () => {
            console.log('连接关闭，5秒后重连...');
            setTimeout(() => this.connect(), 5000);
        });

        console.log('RabbitMQ 连接成功');
    }

    async close() {
        await this.channel?.close();
        await this.connection?.close();
    }
}

module.exports = new RabbitMQClient();
```

## 二、生产者

```javascript
// producer.js
const client = require('./connection');

async function sendOrder(order) {
    const exchange = 'order-exchange';
    const routingKey = 'order.created';

    // 声明交换机
    await client.channel.assertExchange(exchange, 'topic', { durable: true });

    // 发送消息
    const message = Buffer.from(JSON.stringify(order));
    const published = client.channel.publish(exchange, routingKey, message, {
        persistent: true,                    // 消息持久化
        contentType: 'application/json',
        headers: {
            'x-retry-count': 0,
            'x-trace-id': generateTraceId()
        }
    });

    if (!published) {
        console.warn('消息发送失败，信道背压');
    }
}

// 批量发送
async function sendBatch(orders) {
    for (const order of orders) {
        await sendOrder(order);
    }
}

// 发送延迟消息（需要延迟插件）
async function sendDelayed(order, delayMs) {
    const message = Buffer.from(JSON.stringify(order));
    client.channel.publish('delayed-exchange', 'order.delay', message, {
        headers: { 'x-delay': delayMs },
        persistent: true
    });
}
```

## 三、消费者

```javascript
// consumer.js
const client = require('./connection');

async function consumeOrders() {
    const queue = 'order-queue';
    const exchange = 'order-exchange';

    // 声明队列
    await client.channel.assertQueue(queue, {
        durable: true,
        arguments: {
            'x-dead-letter-exchange': 'dlx-exchange',
            'x-max-priority': 10
        }
    });

    // 绑定
    await client.channel.bindQueue(queue, exchange, 'order.#');

    // 设置预取
    await client.channel.prefetch(50);

    // 消费消息
    await client.channel.consume(queue, async (msg) => {
        if (!msg) return;

        try {
            const order = JSON.parse(msg.content.toString());
            console.log('收到订单:', order.id);

            // 处理业务逻辑
            await processOrder(order);

            // 确认消息
            client.channel.ack(msg);

        } catch (error) {
            console.error('处理失败:', error);

            // 拒绝消息，不重新入队（进入死信队列）
            client.channel.nack(msg, false, false);
        }
    }, {
        noAck: false  // 手动确认
    });
}

async function processOrder(order) {
    // 业务处理逻辑
    return new Promise(resolve => setTimeout(resolve, 100));
}

// 启动
(async () => {
    const client = require('./connection');
    await client.connect();
    await consumeOrders();
    console.log('消费者启动成功');
})();
```

## 四、RPC 模式

```javascript
// RPC 客户端
async function rpcCall(queue, message, timeout = 5000) {
    return new Promise(async (resolve, reject) => {
        // 创建临时队列接收响应
        const { queue: replyQueue } = await client.channel.assertQueue('', {
            exclusive: true
        });

        const correlationId = generateUUID();

        // 监听响应
        client.channel.consume(replyQueue, (msg) => {
            if (msg.properties.correlationId === correlationId) {
                resolve(JSON.parse(msg.content.toString()));
            }
        }, { noAck: true });

        // 发送请求
        client.channel.sendToQueue(queue, Buffer.from(JSON.stringify(message)), {
            correlationId,
            replyTo: replyQueue
        });

        // 超时处理
        setTimeout(() => reject(new Error('RPC 超时')), timeout);
    });
}

// RPC 服务端
async function startRPCServer(queue) {
    await client.channel.assertQueue(queue, { durable: true });

    client.channel.consume(queue, async (msg) => {
        const request = JSON.parse(msg.content.toString());
        const result = await handleRequest(request);

        client.channel.sendToQueue(
            msg.properties.replyTo,
            Buffer.from(JSON.stringify(result)),
            { correlationId: msg.properties.correlationId }
        );

        client.channel.ack(msg);
    });
}
```

## 五、注意事项

1. **连接断开后需要重新创建 Channel**，Channel 不可复用
2. **prefetch 设置与 Spring Boot 一致**，控制未确认消息数量
3. **JSON 序列化注意 Date 类型**，传输时转为时间戳或 ISO 字符串
4. **生产环境要做好重连机制**，RabbitMQ 重启后客户端需要自动恢复
5. **单 Channel 不要并发消费**，amqplib 的 Channel 不是线程安全的
