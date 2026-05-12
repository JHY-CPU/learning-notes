# Schema 演进

## 一、安全变更

```graphql
# 安全变更 - 不破坏已有客户端

# 1. 新增可选字段
type User {
  id: ID!
  name: String!
  nickname: String        # 新增 - 安全 ✓
}

# 2. 新增枚举值
enum OrderStatus {
  CREATED
  PAID
  REFUNDING               # 新增 - 安全 ✓
}

# 3. 新增可选参数
type Query {
  users(limit: Int, status: UserStatus): [User!]!
  # 新增 status 参数 - 安全 ✓
}

# 4. 放宽返回类型 (可空化)
type User {
  name: String            # 从 String! 放宽 - 安全 ✓
}
```

## 二、破坏性变更

```graphql
# 破坏性变更 - 会破坏已有客户端

# 1. 删除字段 ✗
type User {
  id: ID!
  # name 字段被删除 - ✗
}

# 2. 重命名字段 ✗
type User {
  id: ID!
  fullName: String!       # 从 name 改为 fullName - ✗
}

# 3. 新增非空字段 ✗
input CreateUserInput {
  name: String!
  email: String!
  phone: String!          # 新增非空 - ✗ 老客户端不会传
}

# 4. 缩小类型 ✗
type User {
  age: Int!               # 从 Float! 缩小 - ✗
}

# 5. 删除枚举值 ✗
enum OrderStatus {
  CREATED
  PAID
  # SHIPPED 被删除 - ✗
}
```

## 三、安全演进策略

```yaml
废弃流程:
  Step 1: 标记废弃
    - 使用 @deprecated 指令
    - 添加 reason 说明

  Step 2: 添加替代
    - 新增替代字段
    - 文档说明迁移方式

  Step 3: 迁移客户端
    - 通知使用方
    - 监控废弃字段使用率

  Step 4: 移除字段
    - 使用率为 0 时移除
    - 保留足够缓冲期
```

```graphql
# 演进示例
type User {
  id: ID!
  name: String! @deprecated(reason: "使用 fullName")
  fullName: String!
  username: String @deprecated(reason: "使用 handle")
  handle: String!
}
```

## 四、Schema 比较工具

```bash
# GraphQL Inspector - CI 集成
npx @graphql-inspector/cli diff \
  old-schema.graphql \
  new-schema.graphql

# 输出示例:
# ┌─────────────────────────────────────────────────┐
# │ Schema Changes                                   │
# ├─────────────────────────────────────────────────┤
# │ FAIL  Field 'User.name' was removed              │
# │ WARN  Field 'User.nickname' was added            │
# │ FAIL  Argument 'status' was added to field       │
# │       'Query.users' (required)                   │
# └─────────────────────────────────────────────────┘
```

```yaml
# CI/CD 集成 (GitHub Actions)
name: Schema Check
on: [pull_request]
jobs:
  check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: npx @graphql-inspector/ci introspect schema.graphql --write schema.graphql
      - run: npx @graphql-inspector/cli diff schema-old.graphql schema-new.graphql --fail-on-breaking
```

## 五、Apollo Schema Registry

```yaml
工作流程:
  1. 开发者提交 Schema 变更
  2. CI 运行 Schema 检查
  3. Schema Registry 比较差异
  4. 检测破坏性变更
  5. 通知受影响的客户端
  6. 合并或修改变更
```

```bash
# 注册 Schema
npx rover subgraph publish \
  my-graph@prod \
  --schema ./schema.graphql \
  --name users
```

## 六、变更审计日志

```graphql
# 在 Resolver 中记录变更
const resolvers = {
  Mutation: {
    updateUser: async (_, { id, input }, context) => {
      const before = await db.users.findById(id);
      const after = await db.users.update(id, input);

      // 记录审计日志
      await auditLog.record({
        entity: 'User',
        entityId: id,
        action: 'UPDATE',
        changes: diff(before, after),
        userId: context.currentUser.id,
        timestamp: new Date(),
      });

      return after;
    },
  },
};
```

## 七、注意事项

1. **永远优先新增而非修改**
2. **废弃字段要有使用率监控**
3. **给客户端足够迁移时间**
4. **破坏性变更需要大版本号**
5. **Schema Registry 帮助追踪变更**
6. **CI 中集成 Schema 比较**，阻止破坏性变更合并
7. **定期清理废弃字段**，使用率为零后移除
