# 实时 GraphQL

## 一、GraphQL Live Queries

```graphql
# 实时查询 - 自动更新
query @live {
  orders(status: PENDING) {
    id
    total
    status
    createdAt
  }
}

# 服务端推送变更
# 当订单状态变化时，自动推送更新
```

```typescript
// graphql-live-query 示例
import { LiveQueryStore } from '@n1ru4l/in-memory-live-query-store';

const liveQueryStore = new LiveQueryStore();

// 数据变更时标记
async function updateOrder(id, input) {
  const order = await orderRepository.update(id, input);
  liveQueryStore.invalidate(`Order:${id}`);
  liveQueryStore.invalidate(`Query.orders`);
  return order;
}
```

## 二、GraphQL Subscriptions 实时推送

```graphql
# 基础实时推送
subscription OnOrderStatusChanged($orderId: ID!) {
  orderStatusChanged(orderId: $orderId) {
    order {
      id
      status
      updatedAt
    }
    previousStatus
  }
}

# 聊天室实时消息
subscription OnNewMessage($roomId: ID!) {
  newMessage(roomId: $roomId) {
    id
    content
    sender {
      id
      name
      avatar
    }
    createdAt
  }
}
```

## 三、GraphQL SSE (Server-Sent Events)

```typescript
// GraphQL SSE - 不需要 WebSocket
import { createHandler } from 'graphql-sse/lib/use/http';

const handler = createHandler({ schema });

// HTTP 处理
http.createServer((req, res) => {
  if (req.url.startsWith('/graphql/stream')) {
    handler(req, res);
  }
});

// 客户端使用
const client = createClient({
  url: '/graphql/stream',
});

// 订阅
const unsubscribe = client.subscribe(
  { query: 'subscription { orderCreated { id total } }' },
  {
    next: (data) => console.log(data),
    error: (err) => console.error(err),
    complete: () => console.log('done'),
  }
);
```

## 四、注意事项

1. **Live Queries 适合数据频繁变更的列表**
2. **Subscriptions 适合事件驱动的推送**
3. **SSE 是 WebSocket 的轻量替代**
4. **生产环境用 Redis PubSub 同步**
5. **实时数据要控制推送频率**
