# Zustand 完全指南

## 1. 安装与基础用法

```bash
npm install zustand
```

### 创建 Store

```tsx
import { create } from 'zustand'

interface BearState {
  bears: number
  increase: () => void
  decrease: () => void
  reset: () => void
}

const useBearStore = create<BearState>()((set) => ({
  bears: 0,
  increase: () => set((state) => ({ bears: state.bears + 1 })),
  decrease: () => set((state) => ({ bears: state.bears - 1 })),
  reset: () => set({ bears: 0 }),
}))
```

### 在组件中使用

```tsx
function BearCounter() {
  const bears = useBearStore((state) => state.bears)
  return <h1>{bears} bears</h1>
}

function Controls() {
  const { increase, decrease, reset } = useBearStore()
  return (
    <div>
      <button onClick={increase}>+</button>
      <button onClick={decrease}>-</button>
      <button onClick={reset}>Reset</button>
    </div>
  )
}

// 也可以获取整个 state（但会导致任何变化都重新渲染）
function BearDisplay() {
  const { bears } = useBearStore()  // 不推荐：整个 state 变化都会重新渲染
  return <div>{bears}</div>
}
```

---

## 2. Selectors 选择器

### 基本 Selector

```tsx
// 只订阅需要的状态片段
const bears = useBearStore((state) => state.bears)
const increase = useBearStore((state) => state.increase)

// 使用多个状态
function BearStats() {
  const bears = useBearStore((state) => state.bears)
  const fish = useBearStore((state) => state.fish)

  // ❌ 每次渲染都创建新对象，导致无限重新渲染
  const { bears, fish } = useBearStore((state) => ({
    bears: state.bears,
    fish: state.fish
  }))

  // ✅ 使用 shallow 比较解决
  const { bears, fish } = useBearStore(
    (state) => ({ bears: state.bears, fish: state.fish }),
    shallow
  )
}
```

### shallow 浅比较

```bash
npm install zustand
# shallow 已内置
```

```tsx
import { create } from 'zustand'
import { shallow } from 'zustand/shallow'

// 当 selector 返回新对象时，需要 shallow 比较
const { name, age } = useBearStore(
  (state) => ({ name: state.name, age: state.age }),
  shallow
)
```

### 自定义比较函数

```tsx
const selected = useBearStore(
  (state) => state.items.filter(item => item.active),
  (prev, next) => prev.length === next.length && prev.every((p, i) => p.id === next[i].id)
)
```

---

## 3. Middleware 中间件

### persist（持久化）

```tsx
import { create } from 'zustand'
import { persist, createJSONStorage } from 'zustand/middleware'

interface SettingsState {
  theme: 'light' | 'dark'
  language: 'zh' | 'en'
  fontSize: number
  setTheme: (theme: 'light' | 'dark') => void
  setLanguage: (lang: 'zh' | 'en') => void
  setFontSize: (size: number) => void
}

const useSettingsStore = create<SettingsState>()(
  persist(
    (set) => ({
      theme: 'light',
      language: 'zh',
      fontSize: 14,
      setTheme: (theme) => set({ theme }),
      setLanguage: (language) => set({ language }),
      setFontSize: (fontSize) => set({ fontSize }),
    }),
    {
      name: 'settings-storage',         // localStorage key
      storage: createJSONStorage(() => localStorage),  // 默认 localStorage
      // storage: createJSONStorage(() => sessionStorage),  // 可选 sessionStorage
      partialize: (state) => ({
        // 只持久化指定字段
        theme: state.theme,
        language: state.language,
      }),
    }
  )
)

// 排除某些字段不持久化
const useAuthStore = create<AuthState>()(
  persist(
    (set) => ({
      user: null,
      token: null,
      isLoading: false,  // 不持久化（在 partialize 中排除）
      login: async (creds) => { /* ... */ },
      logout: () => set({ user: null, token: null }),
    }),
    {
      name: 'auth-storage',
      partialize: (state) => ({
        user: state.user,
        token: state.token,
      }),
    }
  )
)
```

### devtools（Redux DevTools）

```tsx
import { create } from 'zustand'
import { devtools } from 'zustand/middleware'

const useStore = create<StoreState>()(
  devtools(
    (set) => ({
      count: 0,
      increment: () => set((state) => ({ count: state.count + 1 }), false, 'increment'),
      decrement: () => set((state) => ({ count: state.bears - 1 }), false, 'decrement'),
      reset: () => set({ count: 0 }, false, 'reset'),
    }),
    {
      name: 'Counter Store',  // DevTools 中显示的名称
    }
  )
)

// set 的第三个参数是 action 名称，在 DevTools 中显示
set({ count: 0 }, false, 'reset')
```

### immer（不可变更新）

```tsx
import { create } from 'zustand'
import { immer } from 'zustand/middleware/immer'

interface TodoState {
  todos: Todo[]
  addTodo: (text: string) => void
  toggleTodo: (id: string) => void
  removeTodo: (id: string) => void
}

const useTodoStore = create<TodoState>()(
  immer((set) => ({
    todos: [],

    // 使用 immer 可以直接"修改" state
    addTodo: (text) =>
      set((state) => {
        state.todos.push({ id: crypto.randomUUID(), text, done: false })
      }),

    toggleTodo: (id) =>
      set((state) => {
        const todo = state.todos.find((t) => t.id === id)
        if (todo) todo.done = !todo.done
      }),

    removeTodo: (id) =>
      set((state) => {
        const index = state.todos.findIndex((t) => t.id === id)
        if (index !== -1) state.todos.splice(index, 1)
      }),
  }))
)
```

### 组合中间件

```tsx
import { create } from 'zustand'
import { persist, devtools, immer } from 'zustand/middleware'

const useStore = create<StoreState>()(
  devtools(
    persist(
      immer((set) => ({
        // ...
      })),
      { name: 'my-store' }
    ),
    { name: 'My Store' }
  )
)
```

---

## 4. 异步 Actions

```tsx
interface UserState {
  user: User | null
  loading: boolean
  error: string | null
  fetchUser: (id: string) => Promise<void>
  updateUser: (data: Partial<User>) => Promise<void>
}

const useUserStore = create<UserState>()((set) => ({
  user: null,
  loading: false,
  error: null,

  fetchUser: async (id) => {
    set({ loading: true, error: null })
    try {
      const response = await fetch(`/api/users/${id}`)
      if (!response.ok) throw new Error('Failed to fetch')
      const user = await response.json()
      set({ user, loading: false })
    } catch (error) {
      set({ error: (error as Error).message, loading: false })
    }
  },

  updateUser: async (data) => {
    set({ loading: true })
    try {
      const response = await fetch('/api/users', {
        method: 'PATCH',
        body: JSON.stringify(data),
      })
      const user = await response.json()
      set({ user, loading: false })
    } catch (error) {
      set({ error: (error as Error).message, loading: false })
    }
  },
}))
```

---

## 5. 计算值（Derived State）

```tsx
interface CartState {
  items: CartItem[]
  addItem: (item: CartItem) => void
  removeItem: (id: string) => void
}

const useCartStore = create<CartState>()((set) => ({
  items: [],
  addItem: (item) => set((state) => ({ items: [...state.items, item] })),
  removeItem: (id) =>
    set((state) => ({ items: state.items.filter((i) => i.id !== id) })),
}))

// ❌ 不推荐：把计算值存在 store 中
// set((state) => ({
//   items: [...state.items, item],
//   total: state.items.reduce((sum, i) => sum + i.price, 0) + item.price
// }))

// ✅ 推荐：在 selector 中计算
function CartTotal() {
  const total = useCartStore((state) =>
    state.items.reduce((sum, item) => sum + item.price * item.quantity, 0)
  )
  return <div>Total: ¥{total}</div>
}

function CartCount() {
  const count = useCartStore((state) =>
    state.items.reduce((sum, item) => sum + item.quantity, 0)
  )
  return <span>{count}</span>
}
```

### 使用 useMemo 优化复杂计算

```tsx
function ExpensiveComponent() {
  const items = useCartStore((state) => state.items)

  const sortedItems = useMemo(
    () => [...items].sort((a, b) => b.price - a.price),
    [items]
  )

  return (
    <ul>
      {sortedItems.map(item => (
        <li key={item.id}>{item.name}: ¥{item.price}</li>
      ))}
    </ul>
  )
}
```

---

## 6. TypeScript 完整类型

```tsx
import { create } from 'zustand'
import { persist, devtools, immer } from 'zustand/middleware'

// State 类型
interface TodoState {
  todos: Todo[]
  filter: 'all' | 'active' | 'completed'

  // Actions
  addTodo: (text: string) => void
  toggleTodo: (id: string) => void
  removeTodo: (id: string) => void
  setFilter: (filter: TodoState['filter']) => void

  // Computed（通过 selector 获取）
  // 不需要在 interface 中定义
}

// Store 创建
const useTodoStore = create<TodoState>()(
  devtools(
    persist(
      immer((set, get) => ({
        todos: [],
        filter: 'all',

        addTodo: (text) =>
          set((state) => {
            state.todos.push({
              id: crypto.randomUUID(),
              text,
              done: false,
              createdAt: new Date(),
            })
          }),

        toggleTodo: (id) =>
          set((state) => {
            const todo = state.todos.find((t) => t.id === id)
            if (todo) todo.done = !todo.done
          }),

        removeTodo: (id) =>
          set((state) => {
            state.todos = state.todos.filter((t) => t.id !== id)
          }),

        setFilter: (filter) => set({ filter }),
      })),
      { name: 'todo-storage' }
    ),
    { name: 'TodoStore' }
  )
)

// 类型安全的 selector
const useTodoFilter = () =>
  useTodoStore((state) => state.filter)

const useFilteredTodos = () =>
  useTodoStore((state) => {
    switch (state.filter) {
      case 'active': return state.todos.filter((t) => !t.done)
      case 'completed': return state.todos.filter((t) => t.done)
      default: return state.todos
    }
  })
```

---

## 7. 拆分 Store

### 按领域拆分

```tsx
// stores/authStore.ts
interface AuthState {
  user: User | null
  token: string | null
  login: (creds: LoginCredentials) => Promise<void>
  logout: () => void
}
export const useAuthStore = create<AuthState>()(/* ... */)

// stores/cartStore.ts
interface CartState {
  items: CartItem[]
  addItem: (item: CartItem) => void
  clear: () => void
}
export const useCartStore = create<CartState>()(/* ... */)

// stores/uiStore.ts
interface UIState {
  sidebarOpen: boolean
  theme: 'light' | 'dark'
  toggleSidebar: () => void
  setTheme: (theme: 'light' | 'dark') => void
}
export const useUIStore = create<UIState>()(/* ... */)
```

### Store 之间通信

```tsx
// storeA.ts — 在 action 中访问其他 store
import { useAuthStore } from './authStore'

const useCartStore = create<CartState>()((set, get) => ({
  items: [],

  checkout: async () => {
    const token = useAuthStore.getState().token
    if (!token) {
      throw new Error('Not authenticated')
    }
    // 使用 token 进行结账...
  },
}))

// 也可以使用 subscribe 监听其他 store 的变化
useAuthStore.subscribe((state) => {
  if (!state.user) {
    // 用户登出时清空购物车
    useCartStore.getState().clear()
  }
})
```

---

## 8. Zustand vs Redux 对比

| 特性 | Zustand | Redux Toolkit |
|------|---------|---------------|
| 样板代码 | 极少 | 中等 |
| 学习曲线 | 低 | 中等 |
| 包大小 | ~1KB | ~10KB |
| 中间件 | 内置 | 内置 |
| DevTools | 支持 | 原生支持 |
| TypeScript | 简单 | 较好 |
| 持久化 | persist 中间件 | redux-persist |
| 异步 | 原生支持 | thunks/saga |
| 选择器 | 手动 | reselect |

### Zustand 优势
- 无需 Provider 包裹
- 极简 API
- 直接在 store 中写异步逻辑
- 非常小的包体积

### Redux 优势
- 更严格的架构约束
- 更完善的 DevTools
- 更成熟的中间件生态（saga, observable）
- 团队规范更容易统一

---

## 9. 实战：购物车 Store

```tsx
import { create } from 'zustand'
import { persist, devtools, immer } from 'zustand/middleware'

interface Product {
  id: string
  name: string
  price: number
  image: string
}

interface CartItem extends Product {
  quantity: number
}

interface CartState {
  items: CartItem[]
  isOpen: boolean

  // Actions
  addItem: (product: Product) => void
  removeItem: (productId: string) => void
  updateQuantity: (productId: string, quantity: number) => void
  clearCart: () => void
  toggleCart: () => void
}

export const useCartStore = create<CartState>()(
  devtools(
    persist(
      immer((set, get) => ({
        items: [],
        isOpen: false,

        addItem: (product) =>
          set((state) => {
            const existing = state.items.find((i) => i.id === product.id)
            if (existing) {
              existing.quantity += 1
            } else {
              state.items.push({ ...product, quantity: 1 })
            }
          }),

        removeItem: (productId) =>
          set((state) => {
            state.items = state.items.filter((i) => i.id !== productId)
          }),

        updateQuantity: (productId, quantity) =>
          set((state) => {
            if (quantity <= 0) {
              state.items = state.items.filter((i) => i.id !== productId)
            } else {
              const item = state.items.find((i) => i.id === productId)
              if (item) item.quantity = quantity
            }
          }),

        clearCart: () => set({ items: [] }),
        toggleCart: () => set((state) => ({ isOpen: !state.isOpen })),
      })),
      { name: 'cart-storage' }
    ),
    { name: 'CartStore' }
  )
)

// Selectors
export const useCartItems = () => useCartStore((state) => state.items)
export const useCartTotal = () =>
  useCartStore((state) =>
    state.items.reduce((sum, item) => sum + item.price * item.quantity, 0)
  )
export const useCartCount = () =>
  useCartStore((state) =>
    state.items.reduce((sum, item) => sum + item.quantity, 0)
  )
```

---

## 总结

- `create` 创建 store，`set` 更新状态，`get` 获取当前状态
- Selector 选择需要的状态片段，避免不必要的重新渲染
- `persist` 中间件实现持久化，`devtools` 接入调试工具，`immer` 简化嵌套更新
- 异步操作直接写在 store 的 action 中
- 按领域拆分多个 store，通过 `getState()` 互相访问
- Zustand 极简轻量，适合大多数中小型项目
