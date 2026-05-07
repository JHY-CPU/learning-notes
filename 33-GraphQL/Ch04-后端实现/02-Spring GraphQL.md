# Spring GraphQL

## 一、项目配置

```xml
<!-- pom.xml -->
<dependencies>
    <dependency>
        <groupId>org.springframework.boot</groupId>
        <artifactId>spring-boot-starter-graphql</artifactId>
    </dependency>
    <dependency>
        <groupId>org.springframework.boot</groupId>
        <artifactId>spring-boot-starter-web</artifactId>
    </dependency>
</dependencies>
```

```yaml
# application.yml
spring:
  graphql:
    path: /graphql
    graphiql:
      enabled: true            # 开发环境启用 GraphiQL
    schema:
      locations: classpath:graphql/
    websocket:
      path: /graphql           # WebSocket 端点
```

## 二、Schema 文件

```graphql
# src/main/resources/graphql/user.graphqls
type User {
  id: ID!
  name: String!
  email: String!
  orders: [Order!]!
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
```

## 三、Controller 实现

```java
@Controller
public class UserController {

    private final UserService userService;

    @QueryMapping
    public User user(@Argument String id) {
        return userService.findById(id);
    }

    @QueryMapping
    public List<User> users() {
        return userService.findAll();
    }

    @MutationMapping
    public User createUser(@Argument CreateUserInput input) {
        return userService.create(input);
    }

    // 关联字段 - Order 中的 user 字段
    @SchemaMapping(typeName = "Order", field = "user")
    public User userForOrder(Order order) {
        return userService.findById(order.getUserId());
    }

    // User 中的 orders 字段
    @SchemaMapping(typeName = "User")
    public List<Order> orders(User user) {
        return orderService.findByUserId(user.getId());
    }
}
```

## 四、DataLoader 集成

```java
@Controller
public class OrderController {

    @SchemaMapping(typeName = "User")
    public CompletableFuture<List<Order>> orders(
            User user,
            DataLoader<String, List<Order>> ordersLoader) {
        return ordersLoader.load(user.getId());
    }
}

// 注册 DataLoader
@Configuration
public class DataLoaderConfig {

    @Bean
    public DataLoaderRegistrar dataLoaderRegistrar(
            OrderRepository orderRepository) {
        return new DataLoaderRegistrar() {
            @Override
            public void registerDataLoaders(
                    DataLoaderRegistry registry,
                    GraphQLContext context) {
                registry.register("orders",
                    DataLoaderFactory.newDataLoader(
                        userIds -> CompletableFuture.supplyAsync(() ->
                            orderRepository.findByUserIds(userIds)
                        )
                    ));
            }
        };
    }
}
```

## 五、注意事项

1. **Schema 文件放在 resources/graphql/ 目录**
2. **@QueryMapping 自动匹配 Query 入口**
3. **@SchemaMapping 处理关联字段**
4. **DataLoader 注册要配对到 Controller**
5. **GraphiQL 只在开发环境启用**
