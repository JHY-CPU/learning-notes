# Resolver 模式

## 一、Resolver 签名

```typescript
// Resolver 函数签名
type ResolverFn = (
  parent: any,      // 父对象
  args: any,        // 查询参数
  context: any,     // 上下文 (auth, dataloaders)
  info: any         // 查询信息
) => any;

// 示例
const resolvers = {
  Query: {
    user: (parent, { id }, context) => {
      return context.userService.findById(id);
    },
  },
  User: {
    orders: (user, args, context) => {
      return context.orderLoader.load(user.id);
    },
    fullName: (user) => {
      return `${user.firstName} ${user.lastName}`;
    },
  },
};
```

## 二、Resolver 分层

```yaml
Resolver 层次:
  Root Resolver:
    - Query.user
    - Mutation.createUser
    - Subscription.orderCreated

  Field Resolver:
    - User.orders (关联字段)
    - User.fullName (计算字段)
    - Order.items (嵌套对象)

  Nested Resolver:
    - Order.items.product (深层关联)
```

```java
// Java 分层示例
@Controller
public class UserResolver {

    @QueryMapping
    public User user(@Argument String id,
                     @ContextValue UserContext ctx) {
        ctx.requireAuth();
        return userService.findById(id);
    }
}

@Controller
public class UserFieldResolver {

    @SchemaMapping(typeName = "User")
    public CompletableFuture<List<Order>> orders(
            User user,
            DataLoader<String, List<Order>> loader) {
        return loader.load(user.getId());
    }

    @SchemaMapping(typeName = "User")
    public String fullName(User user) {
        return user.getFirstName() + " " + user.getLastName();
    }
}
```

## 三、错误处理

```typescript
import { GraphQLError } from 'graphql';

const resolvers = {
  Mutation: {
    createUser: async (_, { input }, { userService }) => {
      try {
        return await userService.create(input);
      } catch (error) {
        if (error.code === 'DUPLICATE_EMAIL') {
          throw new GraphQLError('邮箱已被使用', {
            extensions: {
              code: 'DUPLICATE_EMAIL',
              http: { status: 400 },
            },
          });
        }
        throw error;
      }
    },
  },
};
```

## 四、注意事项

1. **Resolver 是数据获取的最后一道关**
2. **关联字段用 DataLoader 批量加载**
3. **业务逻辑不要放在 Resolver 中**
4. **错误要规范化抛出 GraphQLError**
5. **异步 Resolver 要返回 Promise**
