# Nuxt3服务端API

## 一、概念说明

Nuxt 3 的 `server/api` 目录用于创建服务端 API 路由，底层由 Nitro 引擎驱动。API 路由运行在服务端，可直接访问数据库、文件系统等，无需单独部署后端服务。文件即路由，与 pages 目录类似。

## 二、具体用法

### 创建 API 路由

```
server/
└── api/
    ├── hello.ts         # → GET /api/hello
    ├── users/
    │   ├── index.ts     # → GET/POST /api/users
    │   └── [id].ts      # → GET/PUT/DELETE /api/users/123
    └── posts.get.ts     # → 仅 GET /api/posts
```

```ts
// server/api/hello.ts
export default defineEventHandler((event) => {
  return {
    message: '你好，这是 Nuxt 3 API',
    timestamp: Date.now()
  }
})
// 访问 GET /api/hello → { "message": "你好，这是 Nuxt 3 API", "timestamp": 1715000000000 }
```

```ts
// server/api/users/index.ts
// 处理 GET 和 POST 请求
export default defineEventHandler(async (event) => {
  const method = getMethod(event)

  if (method === 'GET') {
    // 获取所有用户
    return [
      { id: 1, name: '张三' },
      { id: 2, name: '李四' }
    ]
  }

  if (method === 'POST') {
    // 创建新用户
    const body = await readBody(event)
    // body: { "name": "王五" }
    return { id: 3, ...body, createdAt: new Date().toISOString() }
  }
})
```

### 动态路由参数

```ts
// server/api/users/[id].ts
export default defineEventHandler((event) => {
  const id = getRouterParam(event, 'id')
  // 访问 /api/users/123 → id = '123'

  return {
    id: Number(id),
    name: '用户' + id,
    email: `user${id}@example.com`
  }
})
```

### 请求校验与错误处理

```ts
// server/api/posts.ts
export default defineEventHandler(async (event) => {
  // 读取请求体
  const body = await readBody(event)

  // 参数校验
  if (!body.title || !body.content) {
    throw createError({
      statusCode: 400,
      statusMessage: '标题和内容不能为空'
    })
  }

  // 返回创建结果
  return { id: Date.now(), ...body }
})
// POST /api/posts 带空 body → 400 错误：标题和内容不能为空
```

### 中间件与钩子

```ts
// server/middleware/auth.ts
export default defineEventHandler((event) => {
  // 所有 API 请求都会经过此中间件
  const token = getHeader(event, 'authorization')
  if (event.path?.startsWith('/api/protected')) {
    if (!token) {
      throw createError({ statusCode: 401, statusMessage: '未授权' })
    }
  }
})
```

## 三、注意事项与常见陷阱

1. **API 路由不能在客户端导入**：server 目录下的代码只在服务端运行
2. **文件后缀控制 HTTP 方法**：`.get.ts` 只处理 GET，`.post.ts` 只处理 POST
3. **Nitro 自动处理序列化**：返回普通对象即可，无需手动 JSON.stringify
4. **不要在 API 中使用 Vue 组件或响应式 API**：server 纯粹是 Node.js 环境
5. **开发时 API 热更新**：修改 server/api 下的文件会自动重启服务端，但客户端需要刷新
