# Schema 合并

## 一、模块化 Schema

```graphql
# user.graphqls
type User {
  id: ID!
  name: String!
  email: String!
}

extend type Query {
  user(id: ID!): User
  users: [User!]!
}

extend type Mutation {
  createUser(input: CreateUserInput!): User!
}

# order.graphqls
type Order {
  id: ID!
  total: Float!
  user: User!
}

extend type Query {
  order(id: ID!): Order
  orders(userId: ID!): [Order!]!
}

extend type Mutation {
  createOrder(input: CreateOrderInput!): Order!
}

# schema.graphqls (入口)
type Query {
  _empty: String        # 占位 - extend 要有基础类型
}

type Mutation {
  _empty: String
}
```

## 二、@SchemaMapping 注解方式 (Spring GraphQL)

```java
// 不同的 Controller 处理不同领域的 Schema
@Controller
public class UserController {

    @QueryMapping
    public User user(@Argument String id) {
        return userService.findById(id);
    }

    @QueryMapping
    public List<User> users() {
        return userService.findAll();
    }

    @SchemaMapping(typeName = "Order", field = "user")
    public User userForOrder(Order order) {
        return userService.findById(order.getUserId());
    }
}

@Controller
public class OrderController {

    @QueryMapping
    public Order order(@Argument String id) {
        return orderService.findById(id);
    }

    @MutationMapping
    public Order createOrder(@Argument CreateOrderInput input) {
        return orderService.create(input);
    }
}
```

## 三、Schema 拆分策略

```yaml
拆分方式:
  按领域:
    - user.graphqls
    - order.graphqls
    - product.graphqls

  按功能:
    - query.graphqls
    - mutation.graphqls
    - types.graphqls

  按层级:
    - base.graphqls (基础类型)
    - domain.graphqls (业务类型)
    - api.graphqls (API 入口)
```

## 四、注意事项

1. **extend 要基于已存在的基础类型**
2. **一个 Query/Mutation 入口只定义一次**
3. **模块化按业务领域划分**
4. **Schema 文件名建议用 .graphqls 后缀**
5. **合并后要验证 Schema 完整性**
