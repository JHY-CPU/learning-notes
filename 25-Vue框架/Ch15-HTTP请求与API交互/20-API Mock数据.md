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

## 四、复杂 Mock 场景

```js
// mocks/handlers.js
import { http, HttpResponse, delay } from 'msw'
import { faker } from '@faker-js/faker'

// 生成随机数据
function generateUsers(count) {
  return Array.from({ length: count }, (_, i) => ({
    id: i + 1,
    name: faker.person.fullName(),
    email: faker.internet.email(),
    avatar: faker.image.avatar(),
    createdAt: faker.date.recent().toISOString()
  }))
}

const users = generateUsers(100)

export const handlers = [
  // 分页查询
  http.get('/api/users', ({ request }) => {
    const url = new URL(request.url)
    const page = Number(url.searchParams.get('page') || 1)
    const pageSize = Number(url.searchParams.get('pageSize') || 10)
    const keyword = url.searchParams.get('keyword') || ''

    let filtered = users
    if (keyword) {
      filtered = users.filter(u => u.name.includes(keyword))
    }

    const start = (page - 1) * pageSize
    const list = filtered.slice(start, start + pageSize)

    return HttpResponse.json({
      list,
      total: filtered.length,
      page,
      pageSize
    })
  }),

  // 模拟延迟
  http.get('/api/slow', async () => {
    await delay(2000)
    return HttpResponse.json({ data: 'slow response' })
  }),

  // 模拟随机错误
  http.get('/api/flaky', async () => {
    await delay(500)
    if (Math.random() > 0.7) {
      return HttpResponse.json({ message: '随机失败' }, { status: 500 })
    }
    return HttpResponse.json({ success: true })
  }),

  // 模拟文件上传
  http.post('/api/upload', async ({ request }) => {
    const formData = await request.formData()
    const file = formData.get('file')
    return HttpResponse.json({
      url: `/uploads/${file.name}`,
      size: file.size
    })
  })
]
```

## 五、与 Vitest 集成

```js
// __tests__/api.test.js
import { setupServer } from 'msw/node'
import { handlers } from '../mocks/handlers'

const server = setupServer(...handlers)

beforeAll(() => server.listen())
afterEach(() => server.resetHandlers())
afterAll(() => server.close())

test('获取用户列表', async () => {
  const { data } = await axios.get('/api/users?page=1&pageSize=5')
  expect(data.list).toHaveLength(5)
  expect(data.total).toBe(100)
})
```

## 六、Mock 方案对比

| 方案 | 侵入性 | 真实网络层 | 生产安全 | 学习成本 |
|------|--------|-----------|---------|---------|
| MSW | 零侵入 | 是 | 安全 | 中 |
| json-server | 低 | 是 | 需移除 | 低 |
| 本地 Mock 文件 | 中 | 否 | 需移除 | 低 |
| 代理转发 | 低 | 是 | 安全 | 中 |

## 三、注意事项与常见陷阱

- MSW 拦截的是真实的网络请求，不影响业务代码
- 生产环境不会加载 MSW，不需要条件判断删除 mock
- `public/` 目录下必须有 `mockServiceWorker.js` 文件
- `faker` 库可以生成逼真的随机数据，便于演示和测试
- MSW 也可以用于单元测试（`msw/node`），统一前后端 mock 方案
