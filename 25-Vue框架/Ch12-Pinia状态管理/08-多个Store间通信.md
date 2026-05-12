# 多个Store间通信

## 一、概念说明

Store之间可以互相调用，在一个store的action中使用其他store。这是Pinia的优势之一（Vuex中较复杂）。

```js
// stores/cart.js
import { useUserStore } from './user'

export const useCartStore = defineStore('cart', {
  actions: {
    async checkout() {
      const userStore = useUserStore()
      if (!userStore.isLoggedIn) {
        throw new Error('请先登录')
      }
      // 使用用户信息进行结算
      return api.checkout(this.items, userStore.token)
    }
  }
})
```

## 二、具体用法

### Store间调用示例

```js
// stores/notifications.js
export const useNotifications = defineStore('notifications', () => {
  const items = ref([])
  const add = (msg) => items.value.push({ id: Date.now(), ...msg })
  return { items, add }
})

// stores/auth.js
export const useAuth = defineStore('auth', () => {
  const user = ref(null)
  const isLoggedIn = computed(() => !!user.value)

  const login = async (creds) => {
    user.value = await api.login(creds)
    // 调用其他store
    const notify = useNotifications()
    notify.add({ text: '登录成功', type: 'success' })
  }

  return { user, isLoggedIn, login }
})
```

### 循环依赖处理

```js
// ❌ 避免循环调用
// storeA.action调用storeB.action，storeB.action调用storeA.action

// ✅ 解决方案：提取共享逻辑到独立函数或第三个store
const sharedLogic = {
  validate(data) { /* ... */ }
}

// storeA和storeB都使用sharedLogic
```

## 四、实际项目中的Store通信

```js
// stores/auth.js
export const useAuth = defineStore('auth', () => {
  const token = ref(null)
  const user = ref(null)
  const isLoggedIn = computed(() => !!token.value)

  async function login(creds) {
    const { token: t, user: u } = await api.login(creds)
    token.value = t
    user.value = u
  }

  function logout() {
    token.value = null
    user.value = null

    // 登出时清空其他Store的状态
    const cart = useCartStore()
    cart.clearItems()

    const notifications = useNotifications()
    notifications.add({ text: '已退出登录', type: 'info' })
  }

  return { token, user, isLoggedIn, login, logout }
})
```

## 五、避免循环依赖

```js
// ❌ 循环依赖：A → B → A
// stores/order.js
import { useCart } from './cart'
export const useOrder = defineStore('order', () => {
  const submit = () => {
    const cart = useCart()
    cart.clear()  // cart.clear() 又调用 order 的方法
  }
  return { submit }
})

// ✅ 解决方案1：提取共享逻辑
// utils/orderUtils.js
export function createOrder(cartItems, user) { /* ... */ }

// ✅ 解决方案2：使用事件总线或第三方store
// stores/events.js
export const useEvents = defineStore('events', () => {
  const listeners = ref([])
  const emit = (event) => { /* ... */ }
  return { listeners, emit }
})
```

## 三、注意事项与常见陷阱

1. Store间调用是安全的，Pinia会正确处理依赖
2. 避免Store间形成循环依赖（A调B，B调A）
3. 共享逻辑可提取到公共函数或第三个Store
4. 不要在store的`state`或`getter`中调用其他store（只在action中）
5. 保持Store的单一职责，不要让Store过于耦合
6. 登出时应清理所有相关Store的状态
7. 循环依赖可通过事件总线或提取公共逻辑解决
