# GraphQL与async-graphql

## 一、概念说明

async-graphql 是 Rust 的 GraphQL 实现，支持异步、类型安全的 schema 定义。

```rust
use async_graphql::{Schema, Object, SimpleObject};

#[derive(SimpleObject)]
struct User {
    id: i32,
    name: String,
}

struct Query;

#[Object]
impl Query {
    async fn user(&self, id: i32) -> User {
        User { id, name: "张三".to_string() }
    }
}
```

## 二、具体用法

### 2.1 Schema 定义

```rust
use async_graphql::{Schema, EmptyMutation, EmptySubscription};

let schema = Schema::build(Query, EmptyMutation, EmptySubscription)
    .finish();

// 与 axum 集成
use async_graphql_axum::GraphQL;

let app = Router::new()
    .route("/graphql", GraphQL::new(schema));
```

### 2.2 Mutations

```rust
struct Mutation;

#[Object]
impl Mutation {
    async fn create_user(&self, name: String) -> User {
        User { id: 1, name }
    }
}
```

### 2.3 DataLoader 避免 N+1

```rust
use async_graphql::dataloader::Loader;
use std::collections::HashMap;

struct UserLoader {
    pool: PgPool,
}

#[async_trait]
impl Loader<i32> for UserLoader {
    type Value = User;
    type Error = String;

    async fn load(&self, keys: &[i32]) -> Result<HashMap<i32, User>, Self::Error> {
        let users: Vec<User> = sqlx::query_as!(
            User,
            "SELECT * FROM users WHERE id = ANY($1)",
            keys
        )
        .fetch_all(&self.pool)
        .await
        .map_err(|e| e.to_string())?;

        Ok(users.into_iter().map(|u| (u.id, u)).collect())
    }
}
```

### 2.4 订阅（实时更新）

```rust
use async_graphql::{Subscription, futures_util::Stream};

struct Subscription;

#[Subscription]
impl Subscription {
    async fn messages(&self, ctx: &Context<'_>) -> impl Stream<Item = Message> {
        // 返回消息流
        let rx = ctx.data::<broadcast::Receiver<Message>>().unwrap();
        BroadcastStream::new(rx.resubscribe()).filter_map(|msg| async { msg.ok() })
    }
}
```

## 四、GraphQL vs REST

| 特性 | GraphQL | REST |
|------|---------|------|
| 数据获取 | 按需获取 | 固定端点 |
| 过度获取 | 避免 | 常见 |
| 欠获取 | 避免 | 需要多次请求 |
| 类型系统 | 强类型 | 弱类型 |
| 实时更新 | 订阅 | WebSocket |
| 缓存 | 复杂 | 简单 |

## 五、注意事项与常见陷阱

1. **N+1 问题**：使用 DataLoader 避免批量查询中的 N+1 问题
2. **查询深度限制**：防止恶意深度查询导致性能问题
3. **性能监控**：监控 GraphQL 查询性能，分析慢查询
4. **认证集成**：在 resolver 中验证权限，使用 Context 传递认证信息
5. **文档生成**：GraphQL 自动生成 API 文档，使用 Playground 测试
6. **复杂度分析**：限制查询复杂度，防止资源耗尽
7. **批量处理**：使用批量 DataLoader 减少数据库查询
