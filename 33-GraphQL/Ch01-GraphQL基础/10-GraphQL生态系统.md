# GraphQL 生态系统

## 一、服务端框架

```yaml
Node.js:
  Apollo Server:
    - 最流行
    - 功能全面
    - 生态丰富

  graphql-yoga:
    - 轻量级
    - 性能好
    - Mercurius 作者出品

  Mercurius:
    - Fastify 集成
    - 高性能

Java:
  graphql-java:
    - 底层库
    - 功能强大

  Spring GraphQL:
    - Spring 官方
    - Spring Boot 集成

  DGS (Netflix):
    - Netflix 开源
    - 注解驱动

Python:
  Strawberry:
    - 现代 Python
    - 类型提示驱动

  Ariadne:
    - SDL First
    - 简单易用
```

## 二、客户端库

```yaml
JavaScript/TypeScript:
  Apollo Client:
    - 功能最全
    - 缓存强大
    - React 集成

  urql:
    - 轻量级
    - 可扩展
    - Formidable 出品

  graphql-request:
    - 极简
    - 只负责请求

其他语言:
  Apollo iOS:     Swift/iOS
  Apollo Kotlin:  Kotlin/Android
  graphql-java:   Java
  gql:            Dart/Flutter
```

## 三、工具链

```yaml
开发工具:
  codegen:         # 代码生成
    - TypeScript 类型生成
    - Resolver 类型生成

  graphql-eslint:  # 代码检查
    - Schema 校验
    - 查询校验

  graphql-tools:   # 工具集
    - Schema 合并
    - Mock 数据
    - 指令转换

  graphql-scalars: # 自定义标量
    - DateTime
    - JSON
    - Email
    - URL
```

## 四、代码生成配置

```yaml
# codegen.yml - TypeScript 项目
overwrite: true
schema: "http://localhost:4000/graphql"
documents: "src/**/*.graphql"
generates:
  src/generated/graphql.ts:
    plugins:
      - typescript
      - typescript-operations
      - typescript-react-apollo
      - typescript-resolvers
    config:
      withHooks: true
      withComponent: false
      scalars:
        DateTime: string
        JSON: Record<string, unknown>
```

```bash
# 执行代码生成
npx graphql-codegen --config codegen.yml

# 监听模式
npx graphql-codegen --config codegen.yml --watch
```

生成的 TypeScript 类型:
```typescript
// 自动生成的类型
export type User = {
  __typename?: 'User';
  id: string;
  name: string;
  email: string;
  orders: OrderConnection;
};

export type GetUserQueryVariables = Exact<{
  id: Scalars['ID'];
}>;

export type GetUserQuery = {
  __typename?: 'Query';
  user?: { __typename?: 'User'; id: string; name: string } | null;
};

// 自动生成的 Hook
export function useGetUserQuery(
  baseOptions: Apollo.QueryHookOptions<GetUserQuery, GetUserQueryVariables>
) { /* ... */ }
```

## 五、监控与追踪

```yaml
可观测性工具:
  Apollo Studio:
    - Schema 变更追踪
    - 字段使用统计
    - 查询性能分析
    - 错误率监控

  OpenTelemetry:
    - 链路追踪
    - 字段级耗时
    - 跨服务追踪

  Prometheus + Grafana:
    - QPS 监控
    - 延迟分布
    - 错误率
```

```javascript
// OpenTelemetry 集成
import { ApolloServerPluginUsageReporting } from '@apollo/server/plugin/usageReporting';

const server = new ApolloServer({
  typeDefs,
  resolvers,
  plugins: [
    ApolloServerPluginUsageReporting({
      sendVariableValues: { all: true },
      sendHeaders: { all: true },
    }),
  ],
});
```

## 六、注意事项

1. **Apollo 生态最完善**，适合大多数场景
2. **Spring GraphQL 是 Java Spring 的首选**
3. **codegen 是必备工具**，保证类型安全
4. **工具选择要考虑团队技术栈**
5. **关注 GraphQL Foundation 的标准化进展**
6. **graphql-scalars 提供常用自定义标量**
7. **graphql-eslint 可以在 CI 中校验 Schema 质量**
