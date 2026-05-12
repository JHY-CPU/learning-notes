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

## 四、Mock API调用

```ts
import { setActivePinia, createPinia } from 'pinia'
import { vi } from 'vitest'
import { useAuthStore } from '@/stores/auth'

// Mock API
vi.mock('@/api/auth', () => ({
  login: vi.fn().mockResolvedValue({
    token: 'fake-token',
    user: { id: 1, name: '测试用户' }
  }),
  logout: vi.fn().mockResolvedValue(true)
}))

describe('Auth Store', () => {
  beforeEach(() => setActivePinia(createPinia()))

  it('登录成功', async () => {
    const store = useAuthStore()
    await store.login({ username: 'test', password: '123' })

    expect(store.token).toBe('fake-token')
    expect(store.user.name).toBe('测试用户')
    expect(store.isLoggedIn).toBe(true)
  })

  it('登录失败处理', async () => {
    const { login } = await import('@/api/auth')
    login.mockRejectedValueOnce(new Error('密码错误'))

    const store = useAuthStore()
    await expect(store.login({ username: 'x', password: 'x' }))
      .rejects.toThrow('密码错误')
  })
})
```

## 五、使用createTestingPinia测试组件

```ts
import { mount } from '@vue/test-utils'
import { createTestingPinia } from '@pinia/testing'
import CartButton from '../CartButton.vue'
import { useCartStore } from '@/stores/cart'

describe('CartButton', () => {
  it('点击添加商品', async () => {
    const wrapper = mount(CartButton, {
      global: {
        plugins: [createTestingPinia()]
      }
    })

    const store = useCartStore()

    await wrapper.find('button').trigger('click')

    // 检查 action 是否被调用
    expect(store.addItem).toHaveBeenCalledTimes(1)
  })

  it('显示购物车数量', () => {
    const wrapper = mount(CartButton, {
      global: {
        plugins: [createTestingPinia({
          initialState: {
            cart: { items: [{ id: 1 }, { id: 2 }] }
          }
        })]
      }
    })

    expect(wrapper.text()).toContain('2')
  })
})
```

## 六、测试Store间的依赖

```ts
describe('Order Store', () => {
  beforeEach(() => setActivePinia(createPinia()))

  it('下单时检查登录状态', async () => {
    // 先设置auth store的状态
    const auth = useAuthStore()
    auth.token = 'test-token'

    // 测试order store
    const order = useOrderStore()
    const result = await order.createOrder({ items: [] })

    expect(result).toBeTruthy()
  })

  it('未登录时抛出错误', async () => {
    const order = useOrderStore()
    await expect(order.createOrder({ items: [] }))
      .rejects.toThrow('请先登录')
  })
})
```

## 三、注意事项与常见陷阱

1. 每个测试前调用`setActivePinia(createPinia())`重置
2. 测试异步action时使用`async/await`
3. Mock外部依赖（如API调用）
4. `createTestingPinia`可stub actions避免真实副作用
5. Store测试不需要渲染组件，直接调用即可
6. 测试Store间依赖时需要按顺序设置各个Store的状态
7. `initialState` 可以在测试中直接设置 Store 初始状态
