# Store类型定义

## 一、概念说明

Pinia提供优秀的TypeScript支持。Setup式Store自动推断类型，选项式Store可通过泛型增强类型。

```ts
import { defineStore } from 'pinia'
import { ref, computed } from 'vue'

// Setup式：自动推断类型
export const useUser = defineStore('user', () => {
  const name = ref<string>('张三')
  const age = ref<number>(20)
  const info = computed(() => `${name.value}, ${age.value}岁`)
  const setAge = (val: number) => { age.value = val }
  return { name, age, info, setAge }
})

// 使用时类型完整
const user = useUser()
user.name     // string
user.setAge(25)  // 参数类型检查
```

## 二、具体用法

### 选项式泛型

```ts
interface UserState {
  name: string
  age: number
  token: string | null
}

export const useUser = defineStore<'user', UserState, {
  info: (state: UserState) => string
}, {
  login(creds: { username: string; password: string }): Promise<void>
}>('user', {
  state: () => ({ name: '', age: 0, token: null }),
  getters: {
    info: (state) => `${state.name}, ${state.age}岁`
  },
  actions: {
    async login(creds) { /* ... */ }
  }
})
```

### Store类型导出

```ts
// 导出Store类型
export const useUser = defineStore('user', () => { /* ... */ })

// 在其他地方使用类型
import type { useUser } from '@/stores/user'
type UserStore = ReturnType<typeof useUser>
```

## 四、通用Store类型

```ts
// types/store.ts
import type { Store } from 'pinia'

// 通用Store类型
export type AppStore = Store<string, any>

// Store状态类型
export type StoreState<S> = S extends Store<any, infer T> ? T : never

// 使用
import type { StoreState } from '@/types/store'
import { useUserStore } from '@/stores/user'

type UserState = StoreState<ReturnType<typeof useUserStore>>
// { name: string, age: number, ... }
```

## 五、插件扩展类型

```ts
// plugins/env.d.ts
import 'pinia'

declare module 'pinia' {
  export interface PiniaCustomProperties {
    $env: 'development' | 'production' | 'test'
    $resetAll: () => void
  }

  export interface PiniaCustomStateProperties<S> {
    _lastUpdated: number
  }
}
```

## 六、Setup式Store完整类型示例

```ts
import { defineStore } from 'pinia'
import { ref, computed } from 'vue'

interface UserProfile {
  id: number
  name: string
  email: string
  role: 'admin' | 'user' | 'editor'
}

export const useUserStore = defineStore('user', () => {
  const profile = ref<UserProfile | null>(null)
  const token = ref<string | null>(null)
  const loading = ref(false)

  const isLoggedIn = computed(() => !!token.value)
  const isAdmin = computed(() => profile.value?.role === 'admin')
  const displayName = computed(() => profile.value?.name ?? '未登录')

  async function login(creds: { username: string; password: string }) {
    loading.value = true
    try {
      const res = await api.login(creds)
      token.value = res.token
      profile.value = res.user
    } finally {
      loading.value = false
    }
  }

  function logout() {
    token.value = null
    profile.value = null
  }

  return { profile, token, loading, isLoggedIn, isAdmin, displayName, login, logout }
})

// 类型推断结果：
// profile: Ref<UserProfile | null>
// token: Ref<string | null>
// loading: Ref<boolean>
// isLoggedIn: ComputedRef<boolean>
// login: (creds: { username: string; password: string }) => Promise<void>
```

## 三、注意事项与常见陷阱

1. Setup式Store类型推断最准确，优先使用
2. `ref<T>()`指定泛型确保类型安全
3. `ReturnType<typeof useStore>`获取Store类型
4. 选项式Store的泛型写法较繁琐
5. 插件添加的属性需要扩展类型定义
6. `PiniaCustomProperties` 扩展 store 实例属性
7. `PiniaCustomStateProperties` 扩展 state 中的属性
