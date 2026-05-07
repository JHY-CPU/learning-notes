# API Mock 数据

## 一、概念说明

Mock 数据用于在后端接口未完成时模拟 API 响应，使前端开发不被阻塞。MSW（Mock Service Worker）是目前最优秀的 Mock 方案，它在**网络层**拦截请求，对业务代码零侵入。

```js
// mocks/handlers.js
import { http, HttpResponse } from 'msw'

export const handlers = [
  http.get('/api/users', () => {
    return HttpResponse.json([
      { id: 1, name: '张三' },
      { id: 2, name: '李四' },
    ])
  }),

  http.post('/api/users', async ({ request }) => {
    const body = await request.json()
    return HttpResponse.json({ id: 3, ...body }, { status: 201 })
  }),
]
```

```js
// mocks/browser.js (浏览器端)
import { setupWorker } from 'msw/browser'
import { handlers } from './handlers'

export const worker = setupWorker(...handlers)
```

```js
// main.js
if (import.meta.env.DEV) {
  const { worker } = await import('./mocks/browser')
  await worker.start()
}
```

## 二、具体用法

### 2.1 安装

```bash
npm install msw --save-dev
npx msw init public/ --save
```

### 2.2 模拟延迟和错误

```js
http.get('/api/slow', async () => {
  await delay(2000) // 模拟 2s 延迟
  return HttpResponse.json({ data: 'slow response' })
})

http.get('/api/error', () => {
  return HttpResponse.json(
    { message: '服务器错误' },
    { status: 500 }
  )
})
```

### 2.3 动态路由参数

```js
http.get('/api/users/:id', ({ params }) => {
  return HttpResponse.json({ id: params.id, name: '用户' + params.id })
})
```

## 三、注意事项与常见陷阱

- MSW 拦截的是真实的网络请求，不影响业务代码
- 生产环境不会加载 MSW，不需要条件判断删除 mock
- `public/` 目录下必须有 `mockServiceWorker.js` 文件
