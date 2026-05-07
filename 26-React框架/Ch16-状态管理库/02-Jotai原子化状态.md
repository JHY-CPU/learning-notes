# Jotai 原子化状态管理

## 1. 核心概念

Jotai 采用原子化（Atomic）状态管理模式。与 Zustand 的单一 store 不同，Jotai 将状态拆分为独立的"原子"（Atom），组件只订阅需要的原子，按需组合。

```
Zustand:  Store (包含所有状态) → Selector
Jotai:    Atom A + Atom B + Atom C → 按需组合
```

### 安装

```bash
npm install jotai
```

---

## 2. 基础 Atom

### 创建与使用

```tsx
import { atom, useAtom, useAtomValue, useSetAtom } from 'jotai'

// 创建原子
const countAtom = atom(0)
const nameAtom = atom('Guest')
const itemsAtom = atom<string[]>([])

function Counter() {
  // 方式一：useAtom — 读取和设置
  const [count, setCount] = useAtom(countAtom)

  return (
    <div>
      <span>{count}</span>
      <button onClick={() => setCount(count + 1)}>+1</button>
      <button onClick={() => setCount((c) => c + 1)}>+1 (函数式)</button>
    </div>
  )
}

function Display() {
  // 方式二：useAtomValue — 只读取
  const count = useAtomValue(countAtom)
  return <div>Count: {count}</div>
}

function Controls() {
  // 方式三：useSetAtom — 只设置
  const setCount = useSetAtom(countAtom)

  return (
    <button onClick={() => setCount((c) => c + 1)}>Increment</button>
  )
}
```

### Atom 的优势

```tsx
// 只有使用了 countAtom 的组件会在 count 变化时重新渲染
// 使用了 nameAtom 的组件不会受影响

// 组件 A — 订阅 countAtom
function Counter() {
  const [count, setCount] = useAtom(countAtom)  // count 变化时重新渲染
  // ...
}

// 组件 B — 订阅 nameAtom
function NameInput() {
  const [name, setName] = useAtom(nameAtom)  // count 变化时不会重新渲染
  // ...
}
```

---

## 3. 派生 Atom（Derived Atom）

### 只读派生 Atom

```tsx
const priceAtom = atom(100)
const quantityAtom = atom(5)

// 派生 atom — 只读，自动计算
const totalAtom = atom((get) => {
  const price = get(priceAtom)
  const quantity = get(quantityAtom)
  return price * quantity
})

function TotalDisplay() {
  const total = useAtomValue(totalAtom)  // price 或 quantity 变化时自动更新
  return <div>Total: ¥{total}</div>
}
```

### 可写派生 Atom

```tsx
const countAtom = atom(0)

// 可写的派生 atom — 读和写分离
const doubledCountAtom = atom(
  (get) => get(countAtom) * 2,                    // 读
  (get, set, newValue: number) => {
    set(countAtom, Math.round(newValue / 2))      // 写
  }
)

function DoubledCounter() {
  const [doubled, setDoubled] = useAtom(doubledCountAtom)
  return (
    <div>
      <span>Doubled: {doubled}</span>
      <button onClick={() => setDoubled(10)}>Set doubled to 10</button>
    </div>
  )
}
```

### 复杂派生

```tsx
const usersAtom = atom<User[]>([])
const searchQueryAtom = atom('')

const filteredUsersAtom = atom((get) => {
  const users = get(usersAtom)
  const query = get(searchQueryAtom).toLowerCase()
  return users.filter(user =>
    user.name.toLowerCase().includes(query) ||
    user.email.toLowerCase().includes(query)
  )
})

const activeUsersCountAtom = atom((get) => {
  const users = get(usersAtom)
  return users.filter(u => u.isActive).length
})
```

---

## 4. 异步 Atom

```tsx
// 异步 atom — 返回 Promise
const userIdAtom = atom<string | null>(null)

const userAtom = atom(async (get) => {
  const userId = get(userIdAtom)
  if (!userId) return null

  const response = await fetch(`/api/users/${userId}`)
  if (!response.ok) throw new Error('Failed to fetch')
  return response.json() as Promise<User>
})

function UserProfile() {
  const user = useAtomValue(userAtom)

  // useAtomValue 自动处理 loading/error 状态
  if (user === undefined) return <div>Loading...</div>
  if (user === null) return <div>No user selected</div>

  return <div>{user.name}</div>
}
```

### 使用 Suspense

```tsx
// 异步 atom 配合 Suspense
const dataAtom = atom(async () => {
  const res = await fetch('/api/data')
  return res.json()
})

function App() {
  return (
    <Suspense fallback={<div>Loading...</div>}>
      <DataView />
    </Suspense>
  )
}

function DataView() {
  const data = useAtomValue(dataAtom)  // Suspense 自动处理 loading
  return <div>{JSON.stringify(data)}</div>
}
```

### 可写的异步 Atom

```tsx
const todosAtom = atom<Todo[]>([])

const fetchTodosAtom = atom(
  (get) => get(todosAtom),
  async (_get, set) => {
    const response = await fetch('/api/todos')
    const todos = await response.json()
    set(todosAtom, todos)
  }
)

function TodoList() {
  const todos = useAtomValue(fetchTodosAtom)
  const [, fetchTodos] = useAtom(fetchTodosAtom)

  useEffect(() => {
    fetchTodos()
  }, [fetchTodos])

  return <ul>{todos.map(t => <li key={t.id}>{t.text}</li>)}</ul>
}
```

---

## 5. Atom Family

为每个实例创建独立的原子：

```tsx
import { atomFamily } from 'jotai/utils'

// 根据参数创建 atom 家族
const todoAtomFamily = atomFamily(
  (id: string) => atom<Todo | null>(null),
  (a, b) => a === b  // 可选的比较函数
)

// 使用
function TodoItem({ id }: { id: string }) {
  const todo = useAtomValue(todoAtomFamily(id))
  if (!todo) return null
  return <div>{todo.text}</div>
}

// 带异步的 atomFamily
const userAtomFamily = atomFamily(
  (id: string) => atom(async () => {
    const res = await fetch(`/api/users/${id}`)
    return res.json() as Promise<User>
  })
)
```

---

## 6. 持久化

### atomWithStorage

```tsx
import { atomWithStorage } from 'jotai/utils'

// 自动持久化到 localStorage
const themeAtom = atomWithStorage<'light' | 'dark'>('theme', 'light')
const fontSizeAtom = atomWithStorage('fontSize', 14)
const preferencesAtom = atomWithStorage('preferences', {
  language: 'zh',
  notifications: true,
})

// 使用 sessionStorage
import { atomWithStorage } from 'jotai/utils'

const sessionAtom = atomWithStorage(
  'session',
  null,
  {
    getItem: (key) => {
      const value = sessionStorage.getItem(key)
      return value ? JSON.parse(value) : null
    },
    setItem: (key, value) => {
      sessionStorage.setItem(key, JSON.stringify(value))
    },
    removeItem: (key) => {
      sessionStorage.removeItem(key)
    },
  }
)

function ThemeToggle() {
  const [theme, setTheme] = useAtom(themeAtom)

  return (
    <button onClick={() => setTheme(theme === 'light' ? 'dark' : 'light')}>
      {theme === 'light' ? '🌙' : '☀️'}
    </button>
  )
}
```

### 自定义存储

```tsx
import { atomWithStorage } from 'jotai/utils'

// 创建安全的 JSON 存储（处理 SSR）
function createJSONStorage<T>(key: string, defaultValue: T) {
  return {
    getItem: () => {
      if (typeof window === 'undefined') return defaultValue
      try {
        const value = localStorage.getItem(key)
        return value ? JSON.parse(value) : defaultValue
      } catch {
        return defaultValue
      }
    },
    setItem: (_: string, value: T) => {
      localStorage.setItem(key, JSON.stringify(value))
    },
    removeItem: () => {
      localStorage.removeItem(key)
    },
  }
}
```

---

## 7. DevTools

```tsx
import { useAtomsDevtools } from 'jotai-devtools'  // 单独包
// 或使用 jotai/utils 中的 devtools

// 在应用根部包裹
function App() {
  return (
    <Provider>
      <AtomsDevtools>
        <MyApp />
      </AtomsDevtools>
    </Provider>
  )
}
```

---

## 8. Provider（作用域）

```tsx
import { Provider, atom, useAtom } from 'jotai'

const countAtom = atom(0)

// 不同 Provider 实例拥有独立状态
function App() {
  return (
    <div>
      <Provider>
        <Counter />  {/* 独立状态 A */}
      </Provider>
      <Provider>
        <Counter />  {/* 独立状态 B */}
      </Provider>
    </div>
  )
}
```

---

## 9. Atom in Atom（嵌套模式）

```tsx
const activeTabAtom = atom<string>('home')

// 通过一个 atom 的值来决定另一个 atom 的派生
const tabDataAtom = atom((get) => {
  const tab = get(activeTabAtom)
  switch (tab) {
    case 'home': return get(homeDataAtom)
    case 'settings': return get(settingsDataAtom)
    default: return null
  }
})

// 动态 atom
const selectedTodoIdAtom = atom<string | null>(null)

const selectedTodoAtom = atom((get) => {
  const id = get(selectedTodoIdAtom)
  if (!id) return null
  const todos = get(todosAtom)
  return todos.find(t => t.id === id) ?? null
})
```

---

## 10. TypeScript 完整示例

```tsx
import { atom, useAtom, useAtomValue, useSetAtom } from 'jotai'
import { atomWithStorage } from 'jotai/utils'

// 类型定义
interface User {
  id: string
  name: string
  email: string
}

interface AppState {
  // 不需要，每个 atom 自带类型
}

// 基础 atoms
export const userAtom = atom<User | null>(null)
export const tokenAtom = atomWithStorage<string | null>('token', null)
export const loadingAtom = atom(false)
export const errorAtom = atom<string | null>(null)

// 派生 atoms
export const isAuthenticatedAtom = atom((get) => !!get(tokenAtom))

export const userNameAtom = atom((get) => {
  const user = get(userAtom)
  return user?.name ?? 'Guest'
})

// 可写派生 atom
export const loginAtom = atom(
  null,
  async (get, set, credentials: { email: string; password: string }) => {
    set(loadingAtom, true)
    set(errorAtom, null)

    try {
      const res = await fetch('/api/login', {
        method: 'POST',
        body: JSON.stringify(credentials),
      })

      if (!res.ok) throw new Error('Login failed')

      const { user, token } = await res.json()
      set(userAtom, user)
      set(tokenAtom, token)
    } catch (err) {
      set(errorAtom, (err as Error).message)
    } finally {
      set(loadingAtom, false)
    }
  }
)

// 自定义 Hook
export function useAuth() {
  const user = useAtomValue(userAtom)
  const isAuthenticated = useAtomValue(isAuthenticatedAtom)
  const loading = useAtomValue(loadingAtom)
  const error = useAtomValue(errorAtom)
  const login = useSetAtom(loginAtom)

  return { user, isAuthenticated, loading, error, login }
}
```

---

## 11. Jotai vs Recoil vs Zustand

| 特性 | Jotai | Recoil | Zustand |
|------|-------|--------|---------|
| 模型 | 原子化 | 原子化 | Store |
| 包大小 | ~2.5KB | ~20KB | ~1KB |
| Provider | 可选 | 必需 | 不需要 |
| 异步 | 原生支持 | 原生支持 | 手动处理 |
| 持久化 | atomWithStorage | 第三方 | persist 中间件 |
| SSR | 支持 | 支持 | 支持 |
| 状态 | 活跃维护 | Meta 内部优先 | 活跃维护 |
| 适用场景 | 细粒度状态共享 | 已被 Jotai 取代趋势 | 中小型全局状态 |

### 何时使用 Jotai

- 状态粒度细，需要独立更新
- 很多派生状态（computed）
- 希望按需订阅，避免不必要的渲染
- 状态之间有复杂的依赖关系

### 何时使用 Zustand

- 状态集中在少数几个 store
- 团队熟悉 Redux 模式
- 需要更强的状态组织结构

---

## 总结

- Atom 是 Jotai 的核心概念，每个 atom 是独立的状态单元
- 派生 atom 通过 `get` 其他 atom 实现 computed 状态
- 异步 atom 原生支持，可配合 Suspense
- `atomFamily` 为每个参数创建独立 atom 实例
- `atomWithStorage` 实现自动持久化
- Jotai 适合状态分散、派生复杂的场景
