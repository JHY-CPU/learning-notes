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

## 三、注意事项与常见陷阱

1. Store间调用是安全的，Pinia会正确处理依赖
2. 避免Store间形成循环依赖（A调B，B调A）
3. 共享逻辑可提取到公共函数或第三个Store
4. 不要在store的`state`或`getter`中调用其他store（只在action中）
5. 保持Store的单一职责，不要让Store过于耦合
