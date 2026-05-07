# Store测试

## 一、概念说明

Pinia Store的测试相对简单，因为Store是独立的函数，不依赖DOM。使用`createTestingPinia`进行测试。

```bash
npm install -D @pinia/testing
```

```ts
import { setActivePinia, createPinia } from 'pinia'
import { useCounterStore } from '@/stores/counter'

describe('Counter Store', () => {
  beforeEach(() => {
    setActivePinia(createPinia())  // 每个测试重置
  })

  it('初始值为0', () => {
    const store = useCounterStore()
    expect(store.count).toBe(0)
  })

  it('increment增加计数', () => {
    const store = useCounterStore()
    store.increment()
    expect(store.count).toBe(1)
  })
})
```

## 二、具体用法

### 测试Setup式Store

```ts
describe('Auth Store', () => {
  beforeEach(() => setActivePinia(createPinia()))

  it('登录成功后设置token', async () => {
    const store = useAuthStore()
    await store.login({ username: 'admin', password: '123' })
    expect(store.isLoggedIn).toBe(true)
    expect(store.token).toBeTruthy()
  })

  it('logout清除状态', () => {
    const store = useAuthStore()
    store.logout()
    expect(store.isLoggedIn).toBe(false)
  })
})
```

### 使用createTestingPinia

```ts
import { createTestingPinia } from '@pinia/testing'

const wrapper = mount(MyComponent, {
  global: {
    plugins: [createTestingPinia({
      stubActions: false  // 是否执行真实action
    })]
  }
})
```

## 三、注意事项与常见陷阱

1. 每个测试前调用`setActivePinia(createPinia())`重置
2. 测试异步action时使用`async/await`
3. Mock外部依赖（如API调用）
4. `createTestingPinia`可stub actions避免真实副作用
5. Store测试不需要渲染组件，直接调用即可
