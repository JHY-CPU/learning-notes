# Hooks 类型定义

## 1. useState 类型

### 基本类型推断

```tsx
// TypeScript 自动推断类型
const [count, setCount] = useState(0)           // number
const [name, setName] = useState('')             // string
const [visible, setVisible] = useState(false)    // boolean
const [items, setItems] = useState<string[]>([]) // string[]

// 初始值为 null 时需要显式指定类型
const [user, setUser] = useState<User | null>(null)
const [data, setData] = useState<Data | undefined>(undefined)

// 联合类型状态
type Status = 'idle' | 'loading' | 'success' | 'error'
const [status, setStatus] = useState<Status>('idle')
```

### 函数式更新的类型

```tsx
interface FormState {
  name: string
  email: string
  age: number
}

function Form() {
  const [form, setForm] = useState<FormState>({
    name: '',
    email: '',
    age: 0
  })

  // prev 自动推断为 FormState 类型
  setForm(prev => ({ ...prev, name: 'Alice' }))

  // 通用更新函数
  const updateField = <K extends keyof FormState>(
    field: K,
    value: FormState[K]
  ) => {
    setForm(prev => ({ ...prev, [field]: value }))
  }

  // 事件驱动更新
  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value, type } = e.target
    setForm(prev => ({
      ...prev,
      [name]: type === 'number' ? Number(value) : value
    }))
  }
}
```

---

## 2. useReducer 类型

### 完整的 Action 类型

```tsx
// 定义 State 和 Action
interface CounterState {
  count: number
  step: number
}

type CounterAction =
  | { type: 'increment' }
  | { type: 'decrement' }
  | { type: 'reset' }
  | { type: 'set_count'; payload: number }
  | { type: 'set_step'; payload: number }

// Reducer 函数
function counterReducer(state: CounterState, action: CounterAction): CounterState {
  switch (action.type) {
    case 'increment':
      return { ...state, count: state.count + state.step }
    case 'decrement':
      return { ...state, count: state.count - state.step }
    case 'reset':
      return { ...state, count: 0 }
    case 'set_count':
      return { ...state, count: action.payload }
    case 'set_step':
      return { ...state, step: action.payload }
    default:
      // exhaustive check
      const _exhaustive: never = action
      return state
  }
}

// 使用
function Counter() {
  const [state, dispatch] = useReducer(counterReducer, {
    count: 0,
    step: 1
  })

  return (
    <div>
      <p>Count: {state.count}, Step: {state.step}</p>
      <button onClick={() => dispatch({ type: 'increment' })}>+</button>
      <button onClick={() => dispatch({ type: 'decrement' })}>-</button>
      <button onClick={() => dispatch({ type: 'set_count', payload: 100 })}>
        Set to 100
      </button>
    </div>
  )
}
```

### 使用 as const 和 Action Creator

```tsx
// 使用 as const 自动生成 Action 类型
type Action =
  | { type: 'toggle' }
  | { type: 'set'; payload: boolean }

// Action Creator（辅助函数）
const actions = {
  toggle: () => ({ type: 'toggle' } as const),
  set: (value: boolean) => ({ type: 'set', payload: value } as const)
}

// 从 actions 推断 Action 类型
type InferredAction = ReturnType<typeof actions[keyof typeof actions]>

function reducer(state: boolean, action: InferredAction): boolean {
  switch (action.type) {
    case 'toggle': return !state
    case 'set': return action.payload
  }
}

// 使用
const [active, dispatch] = useReducer(reducer, false)
dispatch(actions.toggle())
dispatch(actions.set(true))
```

### 带异步的 Reducer

```tsx
interface DataState<T> {
  data: T | null
  loading: boolean
  error: string | null
}

type DataAction<T> =
  | { type: 'fetch_start' }
  | { type: 'fetch_success'; payload: T }
  | { type: 'fetch_error'; payload: string }
  | { type: 'reset' }

function dataReducer<T>(state: DataState<T>, action: DataAction<T>): DataState<T> {
  switch (action.type) {
    case 'fetch_start':
      return { ...state, loading: true, error: null }
    case 'fetch_success':
      return { data: action.payload, loading: false, error: null }
    case 'fetch_error':
      return { ...state, loading: false, error: action.payload }
    case 'reset':
      return { data: null, loading: false, error: null }
  }
}

// 使用
function UserProfile({ userId }: { userId: string }) {
  const [state, dispatch] = useReducer(dataReducer<User>, {
    data: null,
    loading: false,
    error: null
  })

  useEffect(() => {
    dispatch({ type: 'fetch_start' })
    fetch(`/api/users/${userId}`)
      .then(res => res.json())
      .then(data => dispatch({ type: 'fetch_success', payload: data }))
      .catch(err => dispatch({ type: 'fetch_error', payload: err.message }))
  }, [userId])
}
```

---

## 3. useRef 类型

### DOM Ref vs Mutable Ref

```tsx
// DOM Ref — 绑定到 DOM 元素
function InputFocus() {
  const inputRef = useRef<HTMLInputElement>(null)

  const focusInput = () => {
    // inputRef.current 可能是 null（组件未挂载时）
    inputRef.current?.focus()
  }

  return (
    <div>
      <input ref={inputRef} type="text" />
      <button onClick={focusInput}>Focus</button>
    </div>
  )
}

// 不同元素类型的 ref
const divRef = useRef<HTMLDivElement>(null)
const buttonRef = useRef<HTMLButtonElement>(null)
const canvasRef = useRef<HTMLCanvasElement>(null)
const videoRef = useRef<HTMLVideoElement>(null)
const formRef = useRef<HTMLFormElement>(null)

// Mutable Ref — 存储任意可变值
function Timer() {
  const intervalRef = useRef<number | null>(null)  // 存储 interval ID
  const countRef = useRef(0)                        // 存储计数（不触发重新渲染）
  const prevValueRef = useRef<string>('')           // 存储上一次的值

  const start = () => {
    intervalRef.current = window.setInterval(() => {
      countRef.current++
      console.log('count:', countRef.current)  // 不会触发重新渲染
    }, 1000)
  }

  const stop = () => {
    if (intervalRef.current !== null) {
      clearInterval(intervalRef.current)
      intervalRef.current = null
    }
  }

  useEffect(() => stop, [])
}
```

### useRef 保存上一次的值

```tsx
function usePrevious<T>(value: T): T | undefined {
  const ref = useRef<T | undefined>(undefined)
  useEffect(() => {
    ref.current = value
  })
  return ref.current
}

function Counter() {
  const [count, setCount] = useState(0)
  const prevCount = usePrevious(count)

  return (
    <div>
      <p>现在: {count}, 之前: {prevCount ?? '无'}</p>
      <button onClick={() => setCount(c => c + 1)}>+1</button>
    </div>
  )
}
```

---

## 4. useContext 类型

### 基本 Context 类型

```tsx
// 定义 Context 值的类型
interface AuthContextType {
  user: User | null
  login: (credentials: LoginCredentials) => Promise<void>
  logout: () => void
  isLoading: boolean
}

// 创建 Context，提供默认值
const AuthContext = createContext<AuthContextType | undefined>(undefined)

// 自定义 Hook 获取 Context
function useAuth(): AuthContextType {
  const context = useContext(AuthContext)
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider')
  }
  return context
}

// Provider 组件
function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<User | null>(null)
  const [isLoading, setIsLoading] = useState(false)

  const login = async (credentials: LoginCredentials) => {
    setIsLoading(true)
    try {
      const user = await api.login(credentials)
      setUser(user)
    } finally {
      setIsLoading(false)
    }
  }

  const logout = () => setUser(null)

  return (
    <AuthContext.Provider value={{ user, login, logout, isLoading }}>
      {children}
    </AuthContext.Provider>
  )
}

// 使用
function Profile() {
  const { user, logout, isLoading } = useAuth()  // 完整类型提示

  if (isLoading) return <Spinner />
  if (!user) return <div>未登录</div>
  return <div>你好，{user.name} <button onClick={logout}>退出</button></div>
}
```

### 使用 null 还是 undefined 作为默认值？

```tsx
// 方式一：undefined + 检查（推荐，更安全）
const ThemeContext = createContext<ThemeContextType | undefined>(undefined)
function useTheme() {
  const ctx = useContext(ThemeContext)
  if (!ctx) throw new Error('...')
  return ctx
}

// 方式二：null + 断言
const ThemeContext = createContext<ThemeContextType | null>(null)
function useTheme() {
  const ctx = useContext(ThemeContext)
  if (!ctx) throw new Error('...')
  return ctx
}

// 方式三：直接提供默认值（不需要检查）
const ThemeContext = createContext<ThemeContextType>({
  theme: 'light',
  toggleTheme: () => {},
})
function useTheme() {
  return useContext(ThemeContext)  // 永远不会是 undefined
}
```

---

## 5. 自定义 Hook 类型

```tsx
// useLocalStorage — 带泛型的自定义 Hook
function useLocalStorage<T>(
  key: string,
  initialValue: T
): [T, (value: T | ((prev: T) => T)) => void] {
  const [storedValue, setStoredValue] = useState<T>(() => {
    try {
      const item = localStorage.getItem(key)
      return item ? JSON.parse(item) : initialValue
    } catch {
      return initialValue
    }
  })

  const setValue = (value: T | ((prev: T) => T)) => {
    setStoredValue(prev => {
      const nextValue = value instanceof Function ? value(prev) : value
      localStorage.setItem(key, JSON.stringify(nextValue))
      return nextValue
    })
  }

  return [storedValue, setValue]
}

// 使用 — T 自动推断
const [name, setName] = useLocalStorage('name', '')           // string
const [count, setCount] = useLocalStorage('count', 0)          // number
const [user, setUser] = useLocalStorage<User | null>('user', null)

// useDebounce
function useDebounce<T>(value: T, delay: number): T {
  const [debouncedValue, setDebouncedValue] = useState(value)

  useEffect(() => {
    const timer = setTimeout(() => setDebouncedValue(value), delay)
    return () => clearTimeout(timer)
  }, [value, delay])

  return debouncedValue
}

// useToggle
function useToggle(initialValue = false): [boolean, () => void, (value: boolean) => void] {
  const [value, setValue] = useState(initialValue)
  const toggle = useCallback(() => setValue(v => !v), [])
  return [value, toggle, setValue]
}

// useFetch — 返回类型化的异步状态
interface UseFetchResult<T> {
  data: T | null
  loading: boolean
  error: Error | null
  refetch: () => void
}

function useFetch<T>(url: string): UseFetchResult<T> {
  const [data, setData] = useState<T | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<Error | null>(null)

  const fetchData = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const res = await fetch(url)
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const json = await res.json()
      setData(json)
    } catch (err) {
      setError(err instanceof Error ? err : new Error(String(err)))
    } finally {
      setLoading(false)
    }
  }, [url])

  useEffect(() => { fetchData() }, [fetchData])

  return { data, loading, error, refetch: fetchData }
}

// 使用
interface User {
  id: number
  name: string
}

function UserList() {
  const { data: users, loading, error } = useFetch<User[]>('/api/users')
  // users 的类型是 User[] | null

  if (loading) return <Spinner />
  if (error) return <div>Error: {error.message}</div>
  return <ul>{users?.map(u => <li key={u.id}>{u.name}</li>)}</ul>
}
```

---

## 6. 泛型 Hooks

```tsx
// useMap — 管理 Map 类型状态
function useMap<K, V>(initialEntries?: Iterable<[K, V]>) {
  const [map, setMap] = useState(() => new Map(initialEntries))

  const actions = useMemo(() => ({
    set: (key: K, value: V) => {
      setMap(prev => {
        const next = new Map(prev)
        next.set(key, value)
        return next
      })
    },
    get: (key: K) => map.get(key),
    has: (key: K) => map.has(key),
    delete: (key: K) => {
      setMap(prev => {
        const next = new Map(prev)
        next.delete(key)
        return next
      })
    },
    clear: () => setMap(new Map()),
    entries: () => Array.from(map.entries()),
    keys: () => Array.from(map.keys()),
    values: () => Array.from(map.values()),
    size: map.size
  }), [map])

  return [map, actions] as const
}

// 使用
function App() {
  const [map, { set, get, has, delete: remove }] = useMap<string, number>()
  set('age', 25)
  const age = get('age')  // number | undefined
}

// useSelection — 管理选择状态
function useSelection<T extends { id: string | number }>(items: T[]) {
  const [selectedIds, setSelectedIds] = useState<Set<T['id']>>(new Set())

  const toggle = (id: T['id']) => {
    setSelectedIds(prev => {
      const next = new Set(prev)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return next
    })
  }

  const selectAll = () => {
    setSelectedIds(new Set(items.map(item => item.id)))
  }

  const clearSelection = () => setSelectedIds(new Set())

  const isSelected = (id: T['id']) => selectedIds.has(id)

  const selectedItems = items.filter(item => selectedIds.has(item.id))

  return {
    selectedIds,
    selectedItems,
    toggle,
    selectAll,
    clearSelection,
    isSelected,
    count: selectedIds.size
  }
}
```

---

## 7. useImperativeHandle 类型

```tsx
// 定义暴露的方法接口
interface InputHandle {
  focus: () => void
  blur: () => void
  clear: () => void
  getValue: () => string
  select: () => void
}

// forwardRef + useImperativeHandle
const FancyInput = forwardRef<InputHandle, {
  label: string
  placeholder?: string
  onChange?: (value: string) => void
}>(function FancyInput({ label, placeholder, onChange }, ref) {
  const inputRef = useRef<HTMLInputElement>(null)
  const [value, setValue] = useState('')

  useImperativeHandle(ref, () => ({
    focus: () => inputRef.current?.focus(),
    blur: () => inputRef.current?.blur(),
    clear: () => {
      setValue('')
      inputRef.current?.focus()
    },
    getValue: () => inputRef.current?.value ?? '',
    select: () => inputRef.current?.select()
  }), [])

  return (
    <div>
      <label>{label}</label>
      <input
        ref={inputRef}
        value={value}
        placeholder={placeholder}
        onChange={(e) => {
          setValue(e.target.value)
          onChange?.(e.target.value)
        }}
      />
    </div>
  )
})

// 使用
function Form() {
  const inputRef = useRef<InputHandle>(null)

  return (
    <div>
      <FancyInput ref={inputRef} label="Name" />
      <button onClick={() => inputRef.current?.focus()}>Focus</button>
      <button onClick={() => inputRef.current?.clear()}>Clear</button>
      <button onClick={() => console.log(inputRef.current?.getValue())}>
        Get Value
      </button>
    </div>
  )
}
```

---

## 8. forwardRef 与泛型

### 基本 forwardRef 类型

```tsx
// forwardRef<RefType, PropsType>
const MyInput = forwardRef<HTMLInputElement, {
  label: string
  error?: string
}>(function MyInput({ label, error, ...props }, ref) {
  return (
    <div>
      <label>{label}</label>
      <input ref={ref} {...props} />
      {error && <span className="error">{error}</span>}
    </div>
  )
})

// 使用
const ref = useRef<HTMLInputElement>(null)
<MyInput ref={ref} label="Name" />
```

### 泛型 forwardRef 组件

```tsx
// 泛型 Select 组件
interface SelectProps<T> {
  options: T[]
  value: T | null
  onChange: (value: T) => void
  getLabel: (item: T) => string
  getKey: (item: T) => string | number
}

// forwardRef 不直接支持泛型，需要类型断言
const GenericSelect = forwardRef(function SelectInner<T>(
  { options, value, onChange, getLabel, getKey }: SelectProps<T>,
  ref: React.ForwardedRef<HTMLSelectElement>
) {
  return (
    <select
      ref={ref}
      value={value ? String(getKey(value)) : ''}
      onChange={(e) => {
        const item = options.find(o => String(getKey(o)) === e.target.value)
        if (item) onChange(item)
      }}
    >
      {options.map(option => (
        <option key={getKey(option)} value={getKey(option)}>
          {getLabel(option)}
        </option>
      ))}
    </select>
  )
}) as <T>(props: SelectProps<T> & { ref?: React.Ref<HTMLSelectElement> }) => JSX.Element

// 使用
const selectRef = useRef<HTMLSelectElement>(null)
<GenericSelect<User>
  ref={selectRef}
  options={users}
  value={selected}
  onChange={setSelected}
  getLabel={u => u.name}
  getKey={u => u.id}
/>
```

### 使用 `<T,>` 语法绕过 JSX 泛型歧义

```tsx
// ❌ <T> 会被解析为 JSX 标签
const Component = <T>(props: Props<T>) => { ... }

// ✅ 使用尾逗号 <T,> 告诉解析器这是泛型
const Component = <T,>(props: Props<T>) => { ... }

// ✅ 在 forwardRef 中
const Select = forwardRef(<T,>(props: SelectProps<T>, ref: ForwardedRef<HTMLSelectElement>) => {
  // ...
})
```

---

## 9. useCallback 与 useMemo 类型

```tsx
// useCallback 类型推断
const handleClick = useCallback(
  (e: React.MouseEvent<HTMLButtonElement>) => {
    console.log(e.currentTarget.value)
  },
  []  // 依赖
)

// useMemo 类型推断
const sortedItems = useMemo(
  () => items.sort((a, b) => a.name.localeCompare(b.name)),
  [items]
)  // 推断为 Item[]

// 显式类型标注
const computedValue = useMemo<{ total: number; average: number }>(() => {
  return {
    total: items.reduce((sum, item) => sum + item.price, 0),
    average: items.reduce((sum, item) => sum + item.price, 0) / items.length
  }
}, [items])
```

---

## 总结

| Hook | 类型要点 |
|------|----------|
| `useState` | null 初始值需显式类型 `<T \| null>(null)` |
| `useReducer` | State + Action 判别联合类型，exhaustive check |
| `useRef` | DOM ref 用 `useRef<T>(null)`，可变 ref 用 `useRef<T>(init)` |
| `useContext` | Context 默认值用 `undefined`，配合自定义 Hook 检查 |
| `useImperativeHandle` | 定义暴露方法的接口类型 |
| `forwardRef` | `forwardRef<RefType, PropsType>`，泛型用 `<T,>` 语法 |
| 自定义 Hook | 利用泛型实现类型复用，返回元组命名清晰 |
