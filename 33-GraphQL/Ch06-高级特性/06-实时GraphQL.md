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

## 四、Apollo Client 订阅

```tsx
import { useSubscription, gql } from '@apollo/client';

const ORDER_STATUS_SUBSCRIPTION = gql`
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
`;

function OrderTracker({ orderId }) {
  const { data, loading, error } = useSubscription(
    ORDER_STATUS_SUBSCRIPTION,
    {
      variables: { orderId },
      onData: ({ data }) => {
        // 新数据到达时的回调
        const newStatus = data.data?.orderStatusChanged?.order?.status;
        if (newStatus === 'DELIVERED') {
          toast.success('订单已送达！');
        }
      },
      onError: (err) => {
        console.error('订阅错误:', err);
      },
    }
  );

  if (loading) return <p>等待更新...</p>;
  if (error) return <p>连接错误</p>;

  return (
    <div>
      <p>当前状态: {data?.orderStatusChanged?.order?.status}</p>
      <p>上一状态: {data?.orderStatusChanged?.previousStatus}</p>
    </div>
  );
}
```

## 五、混合链接配置

```typescript
// WebSocket + HTTP 混合
import { split, HttpLink } from '@apollo/client';
import { getMainDefinition } from '@apollo/client/utilities';
import { GraphQLWsLink } from '@apollo/client/link/subscriptions';
import { createClient } from 'graphql-ws';

const httpLink = new HttpLink({ uri: 'http://localhost:4000/graphql' });

const wsLink = new GraphQLWsLink(
  createClient({ url: 'ws://localhost:4000/graphql' })
);

// 根据操作类型选择链接
const splitLink = split(
  ({ query }) => {
    const definition = getMainDefinition(query);
    return (
      definition.kind === 'OperationDefinition' &&
      definition.operation === 'subscription'
    );
  },
  wsLink,    // 订阅走 WebSocket
  httpLink   // 查询和变更走 HTTP
);

const client = new ApolloClient({
  link: splitLink,
  cache: new InMemoryCache(),
});
```

## 六、聊天室完整示例

```graphql
# 聊天室 Schema
type Message {
  id: ID!
  content: String!
  sender: User!
  room: ChatRoom!
  createdAt: DateTime!
}

type ChatRoom {
  id: ID!
  name: String!
  messages(last: Int): [Message!]!
  members: [User!]!
}

type Mutation {
  sendMessage(roomId: ID!, content: String!): Message!
}

type Subscription {
  newMessage(roomId: ID!): Message!
  memberJoined(roomId: ID!): User!
  memberLeft(roomId: ID!): User!
}
```

## 七、注意事项

1. **Live Queries 适合数据频繁变更的列表**
2. **Subscriptions 适合事件驱动的推送**
3. **SSE 是 WebSocket 的轻量替代**
4. **生产环境用 Redis PubSub 同步**
5. **实时数据要控制推送频率**
6. **split 链接按操作类型选协议**
7. **onData 回调适合触发通知等副作用**
