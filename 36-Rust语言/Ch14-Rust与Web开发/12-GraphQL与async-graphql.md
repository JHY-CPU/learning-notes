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

## 三、注意事项与常见陷阱

1. **N+1 问题**：使用 DataLoader 避免
2. **查询深度限制**：防止恶意查询
3. **性能监控**：监控 GraphQL 查询性能
4. **认证集成**：在 resolver 中验证权限
5. **文档生成**：GraphQL 自动生成 API 文档
