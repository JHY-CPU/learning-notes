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

## 四、注意事项

1. **Apollo 生态最完善**，适合大多数场景
2. **Spring GraphQL 是 Java Spring 的首选**
3. **codegen 是必备工具**，保证类型安全
4. **工具选择要考虑团队技术栈**
5. **关注 GraphQL Foundation 的标准化进展**
