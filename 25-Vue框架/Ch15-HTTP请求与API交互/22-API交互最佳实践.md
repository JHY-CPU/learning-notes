# API 交互最佳实践

## 一、概念说明

API 交互的最佳实践涵盖了错误处理、安全防护、性能优化和代码组织等方面。遵循这些实践可以构建健壮、安全、高效的前后端通信层。

```ts
// utils/request.ts - 完整的请求封装示例
import axios from 'axios'
import { ElMessage } from 'element-plus'
import router from '@/router'

const request = axios.create({
  baseURL: import.meta.env.VITE_API_BASE,
  timeout: 15000,
})

// 请求拦截器
request.interceptors.request.use(config => {
  const token = localStorage.getItem('token')
  if (token) config.headers.Authorization = `Bearer ${token}`
  return config
})

// 响应拦截器
request.interceptors.response.use(
  res => {
    if (res.data.code !== 0) {
      ElMessage.error(res.data.message)
      return Promise.reject(new Error(res.data.message))
    }
    return res.data.data
  },
  error => {
    if (error.response?.status === 401) {
      localStorage.removeItem('token')
      router.push('/login')
    }
    return Promise.reject(error)
  }
)

export default request
```

## 二、具体用法

### 2.1 安全实践

- 使用 HTTPS 加密传输
- Token 存储在 `httpOnly` cookie 中（非 localStorage）
- 所有用户输入在后端校验，前端校验仅提升体验
- 敏感操作添加 CSRF Token

### 2.2 性能实践

- 合理使用缓存，减少重复请求
- 列表数据使用分页或无限滚动
- 图片等资源使用 CDN
- 启用 HTTP/2 和 gzip 压缩

### 2.3 代码组织

```
src/
  api/
    user.ts        # 用户相关接口
    post.ts        # 文章相关接口
  utils/
    request.ts     # Axios 实例和拦截器
  types/
    api.ts         # 接口类型定义
  composables/
    useFetch.ts    # 通用请求 composable
```

### 2.4 环境配置

```env
# .env.development
VITE_API_BASE=http://localhost:3000/api

# .env.production
VITE_API_BASE=https://api.example.com
```

## 三、注意事项与常见陷阱

- 不要在前端存储敏感信息（密码、密钥）
- 错误提示要友好，不要暴露服务器内部信息
- 所有接口都要有 loading 和 error 状态处理
- 写操作后要主动清除相关查询缓存
