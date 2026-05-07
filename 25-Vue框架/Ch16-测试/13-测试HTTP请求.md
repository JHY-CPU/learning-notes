# 测试 HTTP 请求

## 一、概念说明

测试 HTTP 请求有两种主流方案：**MSW（Mock Service Worker）**在网络层拦截请求，适合集成测试；**vi.mock**直接 mock axios 模块，适合单元测试。

```vue
<!-- PostList.vue -->
<script setup>
import { ref, onMounted } from 'vue'
import axios from 'axios'

const posts = ref([])
onMounted(async () => {
  const { data } = await axios.get('/api/posts')
  posts.value = data
})
</script>

<template>
  <ul><li v-for="p in posts" :key="p.id">{{ p.title }}</li></ul>
</template>
```

```js
// 方案一：vi.mock
import { vi, describe, it, expect } from 'vitest'
import { mount, flushPromises } from '@vue/test-utils'
import axios from 'axios'
import PostList from '../PostList.vue'

vi.mock('axios')

describe('PostList', () => {
  it('加载文章列表', async () => {
    axios.get.mockResolvedValue({
      data: [{ id: 1, title: 'Vue 测试' }]
    })

    const wrapper = mount(PostList)
    await flushPromises()

    expect(wrapper.findAll('li')).toHaveLength(1)
    expect(wrapper.text()).toContain('Vue 测试')
  })

  it('请求失败处理', async () => {
    axios.get.mockRejectedValue(new Error('网络错误'))

    const wrapper = mount(PostList)
    await flushPromises()

    expect(wrapper.findAll('li')).toHaveLength(0)
  })
})
```

## 二、具体用法

### 2.1 vi.mock 方式

```js
vi.mock('axios')
axios.get.mockResolvedValue({ data: [] })
axios.post.mockResolvedValue({ data: { id: 1 } })
axios.get.mockRejectedValue(new Error('error'))
```

### 2.2 MSW 方式

```js
// mocks/handlers.js
import { http, HttpResponse } from 'msw'

export const handlers = [
  http.get('/api/posts', () => {
    return HttpResponse.json([{ id: 1, title: '测试文章' }])
  })
]

// tests/setup.js
import { setupServer } from 'msw/node'
import { handlers } from '../mocks/handlers'

const server = setupServer(...handlers)
beforeAll(() => server.listen())
afterEach(() => server.resetHandlers())
afterAll(() => server.close())
```

### 2.3 验证请求参数

```js
await wrapper.find('button').trigger('click')
expect(axios.post).toHaveBeenCalledWith('/api/users', {
  name: '张三'
})
```

## 三、注意事项与常见陷阱

- MSW 更接近真实网络行为，vi.mock 更简单直接
- 每个测试后要重置 mock，避免测试间相互影响
- 验证请求参数时注意深浅拷贝问题
