# 测试 Pinia Store

## 一、概念说明

Pinia 提供了 `createTestingPinia` 工具，可以在测试环境中创建隔离的 Pinia 实例，支持自动模拟 actions、控制初始状态等。

```js
// stores/counter.js
import { defineStore } from 'pinia'

export const useCounterStore = defineStore('counter', {
  state: () => ({ count: 0 }),
  getters: { doubleCount: (s) => s.count * 2 },
  actions: {
    increment() { this.count++ },
    async fetchCount() {
      const res = await fetch('/api/count')
      this.count = (await res.json()).count
    }
  }
})
```

```js
// tests/counter.test.js
import { describe, it, expect, vi } from 'vitest'
import { createTestingPinia } from '@pinia/testing'
import { useCounterStore } from '../stores/counter'
import { mount } from '@vue/test-utils'
import Counter from '../Counter.vue'

describe('Counter Store', () => {
  it('初始状态', () => {
    const pinia = createTestingPinia()
    const store = useCounterStore(pinia)
    expect(store.count).toBe(0)
  })

  it('increment action', () => {
    const pinia = createTestingPinia({ stubActions: false })
    const store = useCounterStore(pinia)
    store.increment()
    expect(store.count).toBe(1)
  })

  it('mock action', () => {
    const pinia = createTestingPinia({ createSpy: vi.fn })
    const store = useCounterStore(pinia)
    store.increment()
    expect(store.increment).toHaveBeenCalled()
  })

  it('组件中使用 store', () => {
    const wrapper = mount(Counter, {
      global: { plugins: [createTestingPinia()] }
    })
    const store = useCounterStore()
    expect(store.count).toBe(0)
  })
})
```

## 二、具体用法

### 2.1 createTestingPinia 配置

```js
createTestingPinia({
  createSpy: vi.fn,           // 自动 mock actions
  stubActions: false,         // 是否存根 actions（false = 执行真实逻辑）
  initialState: {             // 设置初始状态
    counter: { count: 100 }
  },
  fakeApp: true               // 创建模拟 Vue app
})
```

### 2.2 测试 getters

```js
const store = useCounterStore(pinia)
store.count = 5
expect(store.doubleCount).toBe(10)
```

### 2.3 测试异步 action

```js
const store = useCounterStore(pinia)
// stubActions 默认为 true，actions 是 mock 函数
await store.fetchCount()
expect(store.fetchCount).toHaveBeenCalled()
```

## 三、注意事项与常见陷阱

- 每个测试要创建新的 `createTestingPinia()` 实例
- `stubActions: true`（默认）时 actions 不执行真实逻辑
- 组件中使用 `useStore()` 会自动获取测试 pinia 实例
