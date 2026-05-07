# TypeScript 请求类型

## 一、概念说明

在 TypeScript 项目中，为 API 请求和响应定义类型可以提供更好的类型安全和开发体验。通过泛型约束 Axios 响应数据的类型。

```ts
// types/api.ts
export interface User {
  id: number
  name: string
  email: string
  createdAt: string
}

export interface ApiResponse<T> {
  code: number
  message: string
  data: T
}

export interface PaginatedData<T> {
  list: T[]
  total: number
  page: number
  size: number
}
```

```ts
// api/user.ts
import request from '@/utils/request'
import type { User, ApiResponse, PaginatedData } from '@/types/api'

export function getUsers(params: { page: number; size: number }) {
  return request.get<ApiResponse<PaginatedData<User>>>('/api/users', { params })
}

export function getUser(id: number) {
  return request.get<ApiResponse<User>>(`/api/users/${id}`)
}

export function createUser(data: Omit<User, 'id' | 'createdAt'>) {
  return request.post<ApiResponse<User>>('/api/users', data)
}
```

```vue
<script setup lang="ts">
import { ref } from 'vue'
import { getUsers } from '@/api/user'
import type { User } from '@/types/api'

const users = ref<User[]>([])

async function load() {
  const { data } = await getUsers({ page: 1, size: 10 })
  users.value = data.data.list // 完整类型推导
}
</script>
```

## 二、具体用法

### 2.1 Axios 泛型支持

```ts
axios.get<User[]>('/api/users')
axios.post<User>('/api/users', { name: '张三' })
axios.request<User>({ method: 'GET', url: '/api/users' })
```

### 2.2 请求参数类型

```ts
interface QueryParams {
  page?: number
  size?: number
  keyword?: string
}

function getUsers(params: QueryParams) {
  return request.get<ApiResponse<PaginatedData<User>>>('/api/users', { params })
}
```

### 2.3 拦截器类型

```ts
request.interceptors.response.use(
  (response: AxiosResponse<ApiResponse<unknown>>) => {
    if (response.data.code !== 0) {
      return Promise.reject(new Error(response.data.message))
    }
    return response
  }
)
```

## 三、注意事项与常见陷阱

- 为所有 API 函数定义入参和出参类型
- 使用 `Partial<T>`、`Pick<T>`、`Omit<T>` 灵活构建请求参数类型
- 共享类型定义放在 `types/` 目录下，避免重复
