# 项目实战 Docker 微服务架构


## 📦 项目实战 8: Docker 微服务架构


多服务拆分 (auth/user/product/order)、Docker Compose 编排、RabbitMQ 事件通信、Nginx API 网关路由分发。


## 微服务拆分


```
// ========== 服务划分 ==========
//
// ┌──────────┐   ┌───────────┐
// │  Nginx   │   │ RabbitMQ  │
// │ API 网关  │   │ 事件总线   │
// └────┬─────┘   └─────┬─────┘
//      │               │
// ┌────┴─────┐   ┌─────┴──────┐
// │ Auth     │   │ Order      │
// │ :4001    │   │ :4004      │
// ├──────────┤   ├────────────┤
// │ User     │   │ Payment    │
// │ :4002    │   │ :4005      │
// ├──────────┤   ├────────────┤
// │ Product  │   │           │
// │ :4003    │   └────────────┘
// └──────────┘
//
// 每个服务独立数据库:
//   auth_db (PostgreSQL) — 用户认证
//   user_db (PostgreSQL) — 用户资料
//   product_db (MongoDB) — 商品
//   order_db (PostgreSQL) — 订单
//   payment_db (PostgreSQL) — 支付

// ========== 项目目录 ==========
// microservices/
// ├── services/
// │   ├── auth/          # Go
// │   │   ├── main.go
// │   │   ├── handler.go
// │   │   └── Dockerfile
// │   ├── user/          # Node.js
// │   │   ├── src/
// │   │   ├── package.json
// │   │   └── Dockerfile
// │   ├── product/       # FastAPI (Python)
// │   │   ├── app/
// │   │   ├── requirements.txt
// │   │   └── Dockerfile
// │   ├── order/         # Node.js
// │   │   ├── src/
// │   │   └── Dockerfile
// │   └── payment/       # Go
// │       ├── main.go
// │       └── Dockerfile
// ├── gateway/
// │   └── nginx.conf
// ├── docker-compose.yml
// └── .env
```


## Docker Compose 编排


```
# ========== Docker Compose ==========
# docker-compose.yml
version: '3.8'

services:
  # ========== API 网关 ==========
  gateway:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./gateway/nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - auth-service
      - user-service
      - product-service
      - order-service
    networks:
      - public

  # ========== 消息队列 ==========
  rabbitmq:
    image: rabbitmq:3-management-alpine
    environment:
      RABBITMQ_DEFAULT_USER: admin
      RABBITMQ_DEFAULT_PASS: secret
    networks:
      - internal

  # ========== 认证服务 (Go) ==========
  auth-service:
    build: ./services/auth
    environment:
      DB_DSN: "postgres://app:secret@auth-db:5432/auth?sslmode=disable"
      JWT_SECRET: ${JWT_SECRET}
    depends_on:
      auth-db:
        condition: service_healthy
    networks:
      - public
      - internal
    healthcheck:
      test: ["CMD", "wget", "-q", "http://localhost:4001/health"]
      interval: 10s

  auth-db:
    image: postgres:16-alpine
    environment:
      POSTGRES_PASSWORD: secret
      POSTGRES_DB: auth
    volumes:
      - auth-data:/var/lib/postgresql/data
    networks:
      - internal
    healthcheck:
      test: pg_isready -U postgres

  # ========== 商品服务 (FastAPI) ==========
  product-service:
    build: ./services/product
    environment:
      MONGODB_URL: mongodb://product-db:27017
    depends_on:
      product-db:
        condition: service_healthy
    networks:
      - public
      - internal

  product-db:
    image: mongo:7
    volumes:
      - product-data:/data/db
    networks:
      - internal
    healthcheck:
      test: echo 'db.runCommand("ping").ok' | mongosh

  # ========== 订单服务 (Node.js) ==========
  order-service:
    build: ./services/order
    environment:
      DB_DSN: postgres://app:secret@order-db:5432/orders
      RABBITMQ_URL: amqp://admin:secret@rabbbitmq:5672
    depends_on:
      order-db:
        condition: service_healthy
      rabbitmq:
        condition: service_healthy
    networks:
      - public
      - internal

  order-db:
    image: postgres:16-alpine
    environment:
      POSTGRES_PASSWORD: secret
      POSTGRES_DB: orders
    volumes:
      - order-data:/var/lib/postgresql/data
    networks:
      - internal
    healthcheck:
      test: pg_isready -U postgres

networks:
  public:    # 网关可访问
  internal:  # 服务间通信

volumes:
  auth-data:
  product-data:
  order-data:
```


## Nginx 网关与事件通信


```
# ========== Nginx 网关 ==========
# gateway/nginx.conf
events {}

http {
    upstream auth     { server auth-service:4001; }
    upstream user     { server user-service:4002; }
    upstream product  { server product-service:4003; }
    upstream order    { server order-service:4004; }

    server {
        listen 80;

        # 请求体大小限制
        client_max_body_size 10M;

        # 认证服务
        location /api/v1/auth/ {
            proxy_pass http://auth;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
        }

        # 用户服务 (需认证)
        location /api/v1/users/ {
            proxy_pass http://user;
            proxy_set_header Host $host;
            proxy_set_header X-User-ID $http_x_user_id;
        }

        # 商品服务
        location /api/v1/products/ {
            proxy_pass http://product;
        }

        # 订单服务
        location /api/v1/orders/ {
            proxy_pass http://order;
        }

        # 限流
        limit_req zone=api burst=50 nodelay;
    }
}

limit_req_zone $binary_remote_addr zone=api:10m rate=100r/s;

// ========== RabbitMQ 事件通信 ==========
// services/order/src/events.js
// 订单服务监听 & 发布事件

const amqp = require('amqplib');

class EventBus {
  async connect(url) {
    this.conn = await amqp.connect(url);
    this.channel = await this.conn.createChannel();

    // 声明交换机
    await this.channel.assertExchange('order.events', 'topic', { durable: true });

    // 声明队列并绑定
    const q = await this.channel.assertQueue('order.payment', { durable: true });
    await this.channel.bindQueue(q.queue, 'order.events', 'payment.*');
  }

  // 发布事件
  publish(routingKey, data) {
    this.channel.publish('order.events', routingKey,
      Buffer.from(JSON.stringify(data)),
      { persistent: true }
    );
  }

  // 消费事件
  consume(queue, routingKey, handler) {
    this.channel.assertQueue(queue, { durable: true });
    this.channel.bindQueue(queue, 'order.events', routingKey);
    this.channel.consume(queue, async (msg) => {
      try {
        await handler(JSON.parse(msg.content.toString()));
        this.channel.ack(msg);
      } catch (err) {
        this.channel.nack(msg, false, true); // 重新入队
      }
    });
  }
}

// 订单事件流程:
// 1. 订单创建 → publish('order.created', data)
// 2. 支付服务消费 → 处理支付
// 3. 支付完成 → publish('payment.completed', data)
// 4. 库存服务消费 → 扣库存
// 5. 通知服务 → 发送邮件
```


> **Note:** 💡 Docker 微服务要点: 每个服务独立部署+独立数据库; Docker Compose 多服务编排; Nginx 网关路由分发 + 限流; RabbitMQ Topic 交换机事件驱动; 服务健康检查 condition; 网络隔离 public/internal; 环境变量多环境配置; JWT 在网关验证, X-User-ID 传递用户。


## 练习


<!-- Converted from: 7_项目实战 Docker 微服务架构.html -->
