# Pinia最佳实践

## 一、概念说明

遵循最佳实践让Pinia代码更清晰、可维护。主要包括设计原则、命名规范、状态组织等方面。

```js
// ✅ 良好实践示例
export const useUserStore = defineStore('user', () => {
  // 1. 使用ref定义状态
  const profile = ref(null)
  const token = ref(null)

  // 2. 使用computed派生状态
  const isLoggedIn = computed(() => !!token.value)
  const displayName = computed(() => profile.value?.name || '游客')

  // 3. 使用action修改状态（封装业务逻辑）
  const login = async (creds) => {
    const res = await api.login(creds)
    token.value = res.token
    profile.value = res.user
  }

  // 4. 暴露最小接口
  return { profile, token, isLoggedIn, displayName, login }
})
```

## 二、具体用法

### 设计原则

| 原则 | 说明 |
|------|------|
| 单一职责 | 每个Store负责一个业务域 |
| 最小暴露 | 只暴露必要的state和action |
| 封装修改 | 通过action修改state |
| 保持独立 | Store间依赖最小化 |

### 命名规范

```js
// Store命名
useUserStore, useCartStore    // ✅
user, cart                    // ❌ (太简短)

// 文件命名
stores/user.js                // ✅
stores/useUserStore.js        // ❌ (冗余)

// ID命名
defineStore('user', { ... })  // ✅
defineStore('useUser', { ... }) // ❌
```

### 状态组织

```js
// ✅ 扁平化state
state: () => ({
  name: '',
  age: 0,
  address: { city: '', street: '' }
})

// ❌ 过度嵌套
state: () => ({
  user: {
    profile: {
      personal: { name: '', age: 0 }
    }
  }
})
```

## 四、代码审查清单

```
设计层面：
  □ 每个Store是否单一职责
  □ Store命名是否清晰
  □ Store间依赖是否单向
  □ 是否有循环依赖
  □ 暴露的接口是否最小化

状态管理：
  □ state 是否使用函数返回
  □ 大对象是否使用 shallowRef
  □ 不需要响应的数据是否 markRaw
  □ 批量修改是否使用 $patch
  □ 解构是否使用 storeToRefs

异步操作：
  □ loading/error 状态是否管理
  □ 是否处理了竞态条件
  □ 错误是否统一处理
  □ 组件卸载是否取消请求

性能：
  □ 复杂计算是否使用 getter
  □ 是否避免了循环中读取 store
  □ 不需要的 store 是否被导入
  □ 是否有不必要的深度响应式

测试：
  □ 每个 action 是否有测试
  □ 异步操作是否 mock
  □ Store 间依赖是否测试
  □ 错误场景是否覆盖
```

## 五、常见反模式

```js
// ❌ 反模式1：Store过大
export const useAppStore = defineStore('app', () => {
  // 用户、购物车、订单、通知全放一个Store
  const user = ref(null)
  const cart = ref([])
  const orders = ref([])
  const notifications = ref([])
  // ...
})

// ❌ 反模式2：Store中存储DOM
const modalElement = ref(null)

// ❌ 反模式3：Store中存储组件
const sidebarComponent = ref(null)

// ❌ 反模式4：循环依赖
// auth.js 调用 cart.js，cart.js 调用 auth.js

// ❌ 反模式5：过度使用全局状态
const isDropdownOpen = ref(false)  // 应该是局部状态
```

## 三、注意事项与常见陷阱

1. Store数量适中，不要每个组件都创建Store
2. 避免在Store中存储可序列化之外的数据（函数、DOM）
3. 使用`storeToRefs`解构保持响应式
4. 复杂getter使用computed而不是方法
5. 定期审查Store依赖关系，避免循环依赖
6. 代码审查清单可以帮助团队保持代码质量
7. 避免反模式是保持项目长期可维护的关键
