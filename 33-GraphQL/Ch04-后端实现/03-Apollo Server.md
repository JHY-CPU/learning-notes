# Apollo Server

## 一、项目搭建

```typescript
// Apollo Server 4 + Express
import { ApolloServer } from '@apollo/server';
import { expressMiddleware } from '@apollo/server/express4';
import { makeExecutableSchema } from '@graphql-tools/schema';
import express from 'express';
import cors from 'cors';

const typeDefs = `#graphql
  type User {
    id: ID!
    name: String!
    email: String!
  }

  type Query {
    user(id: ID!): User
    users: [User!]!
  }

  type Mutation {
    createUser(input: CreateUserInput!): User!
  }

  input CreateUserInput {
    name: String!
    email: String!
  }
`;

const resolvers = {
  Query: {
    user: (_, { id }) => userService.findById(id),
    users: () => userService.findAll(),
  },
  Mutation: {
    createUser: (_, { input }) => userService.create(input),
  },
};

const server = new ApolloServer({
  schema: makeExecutableSchema({ typeDefs, resolvers }),
});

await server.start();
const app = express();
app.use('/graphql', cors(), express.json(), expressMiddleware(server));
app.listen(4000);
```

## 二、插件系统

```typescript
import { ApolloServerPluginDrainHttpServer } from '@apollo/server/plugin/drainHttpServer';
import { ApolloServerPluginCacheControl } from '@apollo/server/plugin/cacheControl';

const server = new ApolloServer({
  schema,
  plugins: [
    // HTTP 服务优雅关闭
    ApolloServerPluginDrainHttpServer({ httpServer }),

    // 缓存控制
    ApolloServerPluginCacheControl({
      defaultMaxAge: 60,
    }),

    // 自定义插件 - 请求日志
    {
      async requestDidStart() {
        const start = Date.now();
        return {
          async willSendResponse(requestContext) {
            const duration = Date.now() - start;
            console.log(`Query: ${requestContext.request.operationName}, Duration: ${duration}ms`);
          },
        };
      },
    },
  ],
});
```

## 三、Context 和认证

```typescript
const server = new ApolloServer({
  schema,
  context: async ({ req }) => {
    const token = req.headers.authorization?.replace('Bearer ', '');
    const user = token ? await verifyToken(token) : null;
    return { user, loaders: createDataLoaders() };
  },
});

// Resolver 中使用
const resolvers = {
  Query: {
    me: (_, __, { user }) => {
      if (!user) throw new AuthenticationError('未登录');
      return user;
    },
  },
};
```

## 四、注意事项

1. **Apollo Server 4 使用 Express 中间件模式**
2. **插件系统支持生命周期钩子**
3. **Context 每个请求独立创建**
4. **生产环境要用 Apollo Studio 做监控**
5. **GraphQL Schema 和业务逻辑分离**
