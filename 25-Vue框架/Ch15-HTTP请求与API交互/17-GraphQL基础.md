# GraphQL 基础

## 一、概念说明

GraphQL 是一种 API 查询语言，允许客户端**精确指定**需要的数据字段，避免 REST 中的过度获取或不足获取问题。Vue 项目中常配合 Apollo Client 使用。

```vue
<script setup>
import { useQuery, useMutation } from '@vue/apollo-composable'
import gql from 'graphql-tag'

// 查询
const { result, loading, error } = useQuery(gql`
  query GetUsers($limit: Int) {
    users(limit: $limit) {
      id
      name
      email
    }
  }
`, { limit: 10 })

// 变更
const { mutate: createUser } = useMutation(gql`
  mutation CreateUser($name: String!, $email: String!) {
    createUser(name: $name, email: $email) {
      id
      name
    }
  }
`)

async function addUser() {
  await createUser({ name: '张三', email: 'zhangsan@example.com' })
}
</script>

<template>
  <div v-if="loading">加载中...</div>
  <ul v-else>
    <li v-for="user in result?.users" :key="user.id">{{ user.name }}</li>
  </ul>
  <button @click="addUser">添加用户</button>
</template>
```

## 二、具体用法

### 2.1 安装 Apollo Client

```bash
npm install @apollo/client @vue/apollo-composable graphql
```

### 2.2 查询（Query）

```graphql
query {
  users {
    id
    name
    posts { title }
  }
}
```

### 2.3 变更（Mutation）

```graphql
mutation {
  createUser(name: "张三", email: "zhang@example.com") {
    id
    name
  }
}
```

### 2.4 REST vs GraphQL

| 特性 | REST | GraphQL |
|------|------|---------|
| 端点数量 | 多个 | 单一端点 |
| 数据粒度 | 固定 | 客户端指定 |
| 版本管理 | URL 版本号 | Schema 演进 |
| 学习曲线 | 低 | 较高 |

## 三、注意事项与常见陷阱

- GraphQL 需要后端专门支持，不能直接对接 REST 接口
- 注意 N+1 查询问题，后端需使用 DataLoader 优化
- 缓存策略比 REST 复杂，Apollo 提供了内置缓存机制
