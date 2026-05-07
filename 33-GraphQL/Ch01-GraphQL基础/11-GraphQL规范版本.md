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

## 四、注意事项

1. **规范是社区驱动的**，参与 RFC 讨论
2. **Defer/Stream 还在草案阶段**
3. **各实现库对新特性的支持进度不同**
4. **关注 graphql-js 官方实现的更新**
5. **不要使用不稳定的实验特性上生产**
