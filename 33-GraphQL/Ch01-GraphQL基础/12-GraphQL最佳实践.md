# GraphQL 最佳实践

## 一、Schema 设计

```graphql
# ✓ 好的命名
type User {
  id: ID!
  firstName: String!    # camelCase 字段
  lastName: String!
  fullName: String!     # 计算字段
  createdAt: DateTime!  # 时间用 DateTime
}

enum OrderStatus {
  PENDING               # SCREAMING_SNAKE_CASE
  CONFIRMED
  SHIPPED
}

# ✓ Input 命名
input CreateUserInput { }   # Create + 名词 + Input
input UpdateUserInput { }   # Update + 名词 + Input

# ✓ Mutation 命名
type Mutation {
  createUser(input: CreateUserInput!): User!
  updateUser(id: ID!, input: UpdateUserInput!): User!
  deleteUser(id: ID!): DeleteResult!
}

# ✓ 返回类型
type DeleteResult {
  success: Boolean!
  message: String
}
```

## 二、错误处理

```graphql
# ✓ 自定义错误扩展
{
  "errors": [
    {
      "message": "用户不存在",
      "locations": [{ "line": 2, "column": 3 }],
      "path": ["user"],
      "extensions": {
        "code": "USER_NOT_FOUND",
        "userId": "999",
        "timestamp": "2024-01-01T00:00:00Z"
      }
    }
  ]
}
```

```java
// Java 错误处理
@ExceptionHandler(GraphQLError.class)
public GraphQLError handle(GraphQLException ex) {
    return GraphQLError.newError()
        .message(ex.getMessage())
        .extensions(Map.of(
            "code", ex.getErrorCode(),
            "timestamp", Instant.now()
        ))
        .build();
}
```

## 三、安全实践

```yaml
安全措施:
  查询深度限制:
    - 防止深度嵌套攻击
    - 建议: maxDepth: 10

  查询复杂度限制:
    - 防止资源耗尽
    - 建议: maxComplexity: 1000

  速率限制:
    - 按用户/IP 限流
    - 认证请求独立限流

  自省禁用:
    - 生产环境关闭
    - 防止 Schema 泄露

  字段级授权:
    - 敏感字段需权限
    - @auth 指令控制
```

## 四、注意事项

1. **Schema 是契约**，变更要审慎
2. **错误码要规范化**
3. **查询限制必须配置**
4. **生产环境关闭自省和 Playground**
5. **日志要记录慢查询和错误查询**
