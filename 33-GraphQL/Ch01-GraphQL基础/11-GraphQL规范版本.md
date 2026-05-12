# GraphQL 规范版本

## 一、规范演进

```yaml
GraphQL 规范:
  2015: Facebook 开源 GraphQL
  2018: GraphQL Foundation 成立
  2021: 规范正式发布为 Open Spec

核心规范:
  - 查询语言
  - 类型系统
  - 执行算法
  - 响应格式
  - 内省系统

RFC 提案:
  - @oneOf - 互斥输入
  - 输入联合类型
  - 定义指令位置
  - 流式传输 (Defer/Stream)
```

## 二、Defer 和 Stream

```graphql
# @defer - 延迟加载字段
query GetUser {
  user(id: "1") {
    name
    email
    ... @defer {
      orders {
        id
        total
      }
    }
  }
}

# 响应流式返回
# 第一个响应 - 快速数据
{ "data": { "user": { "name": "张三", "email": "..." } } }
# 第二个响应 - 延迟数据
{ "data": { "user": { "orders": [...] } }, "path": ["user"] }

# @stream - 流式列表
query GetProducts {
  products {
    id
    name
    reviews @stream(initialCount: 10) {
      rating
      comment
    }
  }
}
```

## 三、@oneOf 提案

```graphql
# @oneOf - 互斥输入
input UserFilter @oneOf {
  id: ID
  email: String
  username: String
}

# 使用时只能传一个字段
query {
  user(filter: { id: "1" }) { name }       # ✓
  user(filter: { email: "a@b.com" }) { name } # ✓
  user(filter: { id: "1", email: "a@b.com" }) # ✗ 错误
}
```

## 四、Defer/Stream 客户端处理

```javascript
// Apollo Client 处理 @defer 响应
import { useQuery, gql } from '@apollo/client';

const GET_USER = gql`
  query GetUser {
    user(id: "1") {
      name
      email
      ... @defer {
        orders {
          id
          total
        }
      }
    }
  }
`;

function UserProfile() {
  const { data, loading } = useQuery(GET_USER);

  // 初始渲染时 orders 为 undefined
  // 后续增量数据到达后自动更新
  return (
    <div>
      <h1>{data?.user.name}</h1>
      <p>{data?.user.email}</p>
      {data?.user.orders ? (
        <OrderList orders={data.user.orders} />
      ) : (
        <p>加载订单中...</p>
      )}
    </div>
  );
}
```

## 五、@stream 服务端实现

```javascript
// Apollo Server 4 支持
import { ApolloServer } from '@apollo/server';
import { expressMiddleware } from '@apollo/server/express4';

// 启用实验性 Defer/Stream
const server = new ApolloServer({
  schema,
  // Apollo Server 4 内置支持 incremental delivery
});

// 客户端流式接收
// 响应格式:
// ---boundary
// Content-Type: application/json
// { "data": { "products": [{ "id": "1" }] } }
// ---boundary
// Content-Type: application/json
// { "incremental": [{ "data": { "reviews": [...] }, "path": ["products", 0] }] }
// ---boundary--
```

## 六、RFC 提案状态

```yaml
GraphQL 规范 RFC 状态:
  已接受 (Accepted):
    - @oneOf: 互斥输入类型
    - 输入联合类型 (Input Union)

  草案 (Draft):
    - @defer / @stream: 增量数据传输
    - 定义指令位置 (Directive Definition Location)

  已实现 (Implemented):
    - 自定义标量
    - 接口实现多个接口 (多接口 implements)

  提议 (Proposed):
    - 全局 ID 标准化
    - Schema 描述标准化
    - Null 值可选性
```

## 七、注意事项

1. **规范是社区驱动的**，参与 RFC 讨论
2. **Defer/Stream 还在草案阶段**
3. **各实现库对新特性的支持进度不同**
4. **关注 graphql-js 官方实现的更新**
5. **不要使用不稳定的实验特性上生产**
6. **@oneOf 已被接受**，部分库已实现
7. **定期检查 graphql-js release notes 获取最新支持状态**
