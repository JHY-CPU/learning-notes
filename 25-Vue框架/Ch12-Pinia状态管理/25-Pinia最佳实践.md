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

## 三、注意事项与常见陷阱

1. Store数量适中，不要每个组件都创建Store
2. 避免在Store中存储可序列化之外的数据（函数、DOM）
3. 使用`storeToRefs`解构保持响应式
4. 复杂getter使用computed而不是方法
5. 定期审查Store依赖关系，避免循环依赖
