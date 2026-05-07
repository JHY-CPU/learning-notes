# Redux Toolkit 深入

## 1. Redux Toolkit 基础回顾

### 安装

```bash
npm install @reduxjs/toolkit react-redux
npm install -D @types/react-redux
```

### Store 配置

```ts
// store/index.ts
import { configureStore } from '@reduxjs/toolkit'
import { TypedUseSelectorHook, useDispatch, useSelector } from 'react-redux'

export const store = configureStore({
  reducer: {
    // 各 slice 的 reducer
    auth: authReducer,
    cart: cartReducer,
    ui: uiReducer,
  },
  // middleware 可以自定义，默认包含 thunk 和 serialization 检查
  middleware: (getDefaultMiddleware) =>
    getDefaultMiddleware({
      serializableCheck: {
        ignoredActions: ['persist/PERSIST'],  // 忽略 redux-persist 的 action
      },
    }),
  devTools: process.env.NODE_ENV !== 'production',
})

// 推导类型
export type RootState = ReturnType<typeof store.getState>
export type AppDispatch = typeof store.dispatch

// 类型化的 hooks
export const useAppDispatch = () => useDispatch<AppDispatch>()
export const useAppSelector: TypedUseSelectorHook<RootState> = useSelector
```

---

## 2. createSlice 深入

```tsx
import { createSlice, PayloadAction, nanoid } from '@reduxjs/toolkit'

interface Todo {
  id: string
  text: string
  completed: boolean
  createdAt: string
}

interface TodoState {
  todos: Todo[]
  filter: 'all' | 'active' | 'completed'
  loading: boolean
}

const initialState: TodoState = {
  todos: [],
  filter: 'all',
  loading: false,
}

const todoSlice = createSlice({
  name: 'todos',
  initialState,
  reducers: {
    // 简单 action
    addTodo: {
      // prepare 回调：自定义 action payload
      reducer: (state, action: PayloadAction<Todo>) => {
        state.todos.push(action.payload)
      },
      prepare: (text: string) => ({
        payload: {
          id: nanoid(),
          text,
          completed: false,
          createdAt: new Date().toISOString(),
        } as Todo,
      }),
    },

    // 使用 Immer 直接修改 state
    toggleTodo: (state, action: PayloadAction<string>) => {
      const todo = state.todos.find((t) => t.id === action.payload)
      if (todo) {
        todo.completed = !todo.completed
      }
    },

    removeTodo: (state, action: PayloadAction<string>) => {
      state.todos = state.todos.filter((t) => t.id !== action.payload)
    },

    editTodo: (state, action: PayloadAction<{ id: string; text: string }>) => {
      const todo = state.todos.find((t) => t.id === action.payload.id)
      if (todo) {
        todo.text = action.payload.text
      }
    },

    // 批量操作
    markAllCompleted: (state) => {
      state.todos.forEach((todo) => {
        todo.completed = true
      })
    },

    clearCompleted: (state) => {
      state.todos = state.todos.filter((t) => !t.completed)
    },

    setFilter: (state, action: PayloadAction<TodoState['filter']>) => {
      state.filter = action.payload
    },

    // 配合 extraReducers 的 loading 状态（也可以在 extraReducers 中处理）
    setLoading: (state, action: PayloadAction<boolean>) => {
      state.loading = action.payload
    },
  },
})

export const {
  addTodo,
  toggleTodo,
  removeTodo,
  editTodo,
  markAllCompleted,
  clearCompleted,
  setFilter,
} = todoSlice.actions

export default todoSlice.reducer
```

---

## 3. extraReducers

用于响应其他 slice 的 action 或 createAsyncThunk：

```tsx
import { createSlice, createAsyncThunk } from '@reduxjs/toolkit'

// Thunk
const fetchTodos = createAsyncThunk(
  'todos/fetchTodos',
  async () => {
    const response = await fetch('/api/todos')
    return response.json()
  }
)

const todoSlice = createSlice({
  name: 'todos',
  initialState,
  reducers: {
    // 同步 reducers
  },
  extraReducers: (builder) => {
    builder
      // 处理 thunk 的 pending 状态
      .addCase(fetchTodos.pending, (state) => {
        state.loading = true
        state.error = null
      })
      // 处理 thunk 的 fulfilled 状态
      .addCase(fetchTodos.fulfilled, (state, action) => {
        state.loading = false
        state.todos = action.payload
      })
      // 处理 thunk 的 rejected 状态
      .addCase(fetchTodos.rejected, (state, action) => {
        state.loading = false
        state.error = action.error.message ?? 'Unknown error'
      })
      // 响应其他 slice 的 action
      .addCase(authSlice.actions.logout, (state) => {
        state.todos = []  // 用户登出时清空 todo
      })
      // matcher：匹配多个 action
      .addMatcher(
        (action) => action.type.startsWith('todos/') && action.type.endsWith('/pending'),
        (state) => { state.loading = true }
      )
      // defaultCase
      .addDefaultCase((state) => {
        // 未匹配的 action
      })
  },
})
```

---

## 4. createAsyncThunk

```tsx
import { createAsyncThunk, createSlice } from '@reduxjs/toolkit'

// 基础用法
export const fetchUser = createAsyncThunk(
  'users/fetchById',
  async (userId: string, { rejectWithValue }) => {
    try {
      const response = await fetch(`/api/users/${userId}`)
      if (!response.ok) {
        const error = await response.json()
        return rejectWithValue(error.message)  // 自定义 rejection 值
      }
      return await response.json()
    } catch (err) {
      return rejectWithValue((err as Error).message)
    }
  },
  {
    // 条件：如果已经在加载，则跳过
    condition: (userId, { getState }) => {
      const { users } = getState() as RootState
      if (users.loading) return false  // 取消这次请求
    },
  }
)

// 带参数的 POST 请求
export const createPost = createAsyncThunk(
  'posts/create',
  async (post: { title: string; body: string }, { getState }) => {
    const { auth } = getState() as RootState
    const response = await fetch('/api/posts', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${auth.token}`,
      },
      body: JSON.stringify(post),
    })
    return response.json()
  }
)

// 在 slice 中使用
const userSlice = createSlice({
  name: 'users',
  initialState: {
    entities: {} as Record<string, User>,
    loading: false,
    error: null as string | null,
  },
  reducers: {},
  extraReducers: (builder) => {
    builder
      .addCase(fetchUser.pending, (state) => {
        state.loading = true
      })
      .addCase(fetchUser.fulfilled, (state, action) => {
        state.loading = false
        state.entities[action.payload.id] = action.payload
      })
      .addCase(fetchUser.rejected, (state, action) => {
        state.loading = false
        state.error = action.payload as string
      })
  },
})
```

### 使用 dispatch 调用 thunk

```tsx
function UserPage({ userId }: { userId: string }) {
  const dispatch = useAppDispatch()
  const user = useAppSelector((state) => state.users.entities[userId])
  const loading = useAppSelector((state) => state.users.loading)

  useEffect(() => {
    dispatch(fetchUser(userId))
  }, [dispatch, userId])

  if (loading) return <Spinner />
  if (!user) return <div>User not found</div>
  return <div>{user.name}</div>
}

// 在组件中直接 dispatch
function CreateButton() {
  const dispatch = useAppDispatch()

  const handleCreate = async () => {
    try {
      const result = await dispatch(createPost({ title: 'New', body: '...' })).unwrap()
      // unwrap() 提取 fulfilled 的 payload 或抛出 rejected 的值
      console.log('Created:', result)
    } catch (err) {
      console.error('Failed:', err)
    }
  }
}
```

---

## 5. RTK Query

RTK Query 是 Redux Toolkit 内置的数据获取和缓存解决方案。

### 基本配置

```tsx
// services/api.ts
import { createApi, fetchBaseQuery } from '@reduxjs/toolkit/query/react'

export const api = createApi({
  reducerPath: 'api',
  baseQuery: fetchBaseQuery({
    baseUrl: '/api',
    prepareHeaders: (headers, { getState }) => {
      const token = (getState() as RootState).auth.token
      if (token) {
        headers.set('Authorization', `Bearer ${token}`)
      }
      return headers
    },
  }),
  tagTypes: ['User', 'Post', 'Comment'],  // 缓存标签
  endpoints: () => ({}),
})

// store 配置
import { configureStore } from '@reduxjs/toolkit'
import { api } from './services/api'

export const store = configureStore({
  reducer: {
    [api.reducerPath]: api.reducer,
    auth: authReducer,
  },
  middleware: (getDefaultMiddleware) =>
    getDefaultMiddleware().concat(api.middleware),
})
```

### 定义 Endpoints

```tsx
// services/userApi.ts
import { api } from './api'

interface User {
  id: string
  name: string
  email: string
}

interface CreateUserRequest {
  name: string
  email: string
  password: string
}

const userApi = api.injectEndpoints({
  endpoints: (builder) => ({
    // GET — 查询
    getUsers: builder.query<User[], void>({
      query: () => '/users',
      providesTags: (result) =>
        result
          ? [
              ...result.map(({ id }) => ({ type: 'User' as const, id })),
              { type: 'User', id: 'LIST' },
            ]
          : [{ type: 'User', id: 'LIST' }],
    }),

    // GET by ID
    getUser: builder.query<User, string>({
      query: (id) => `/users/${id}`,
      providesTags: (result, error, id) => [{ type: 'User', id }],
    }),

    // POST — 创建
    createUser: builder.mutation<User, CreateUserRequest>({
      query: (body) => ({
        url: '/users',
        method: 'POST',
        body,
      }),
      invalidatesTags: [{ type: 'User', id: 'LIST' }],
    }),

    // PUT — 更新
    updateUser: builder.mutation<User, Partial<User> & { id: string }>({
      query: ({ id, ...patch }) => ({
        url: `/users/${id}`,
        method: 'PATCH',
        body: patch,
      }),
      invalidatesTags: (result, error, { id }) => [{ type: 'User', id }],
    }),

    // DELETE
    deleteUser: builder.mutation<void, string>({
      query: (id) => ({
        url: `/users/${id}`,
        method: 'DELETE',
      }),
      invalidatesTags: (result, error, id) => [{ type: 'User', id }],
    }),
  }),
})

export const {
  useGetUsersQuery,
  useGetUserQuery,
  useCreateUserMutation,
  useUpdateUserMutation,
  useDeleteUserMutation,
} = userApi
```

### 在组件中使用

```tsx
// Query Hook — 自动缓存和重新获取
function UserList() {
  const { data: users, error, isLoading, isFetching, refetch } = useGetUsersQuery()

  if (isLoading) return <Spinner />
  if (error) return <div>Error loading users</div>

  return (
    <div>
      <button onClick={refetch}>刷新</button>
      {isFetching && <span>更新中...</span>}
      <ul>
        {users?.map(user => (
          <li key={user.id}>{user.name}</li>
        ))}
      </ul>
    </div>
  )
}

// Mutation Hook — 修改数据
function CreateUserForm() {
  const [createUser, { isLoading }] = useCreateUserMutation()
  const [form, setForm] = useState({ name: '', email: '', password: '' })

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    try {
      await createUser(form).unwrap()
      // 成功后自动重新获取 getUsers
      setForm({ name: '', email: '', password: '' })
    } catch (err) {
      console.error('Failed to create user:', err)
    }
  }

  return (
    <form onSubmit={handleSubmit}>
      <input value={form.name} onChange={e => setForm({ ...form, name: e.target.value })} />
      <input value={form.email} onChange={e => setForm({ ...form, email: e.target.value })} />
      <button type="submit" disabled={isLoading}>
        {isLoading ? 'Creating...' : 'Create'}
      </button>
    </form>
  )
}
```

### 缓存标签（Tags）

```
providesTags: 返回数据提供的标签
invalidatesTags: 操作后失效的标签

逻辑：
  getUsers → providesTags: ['User/LIST', 'User/1', 'User/2']
  createUser → invalidatesTags: ['User/LIST']  → 触发 getUsers 重新获取
  updateUser(1) → invalidatesTags: ['User/1'] → 触发 getUser(1) 重新获取
```

### 高级功能

```tsx
// Polling — 轮询
const { data } = useGetUserQuery(id, {
  pollingInterval: 30000,      // 每 30 秒重新获取
  skipPollingIfUnfocused: true, // 窗口失焦时停止
})

// Lazy Query — 手动触发
const [trigger, { data, isFetching }] = useLazyGetUserQuery()
const handleClick = () => trigger(userId)

// Optimistic Updates — 乐观更新
const updateUser = builder.mutation<User, Partial<User>>({
  query: ({ id, ...patch }) => ({
    url: `/users/${id}`,
    method: 'PATCH',
    body: patch,
  }),
  async onQueryStarted({ id, ...patch }, { dispatch, queryFulfilled }) {
    // 乐观更新：立即更新缓存
    const patchResult = dispatch(
      userApi.util.updateQueryData('getUser', id, (draft) => {
        Object.assign(draft, patch)
      })
    )
    try {
      await queryFulfilled  // 等待请求完成
    } catch {
      patchResult.undo()    // 失败时回滚
    }
  },
})
```

---

## 6. Normalized State（规范化状态）

### createEntityAdapter

```tsx
import { createEntityAdapter, createSlice, EntityState } from '@reduxjs/toolkit'

interface Comment {
  id: string
  postId: string
  author: string
  text: string
  createdAt: string
}

// 创建 entity adapter
const commentsAdapter = createEntityAdapter<Comment>({
  // 指定 ID 字段（默认使用 id）
  selectId: (comment) => comment.id,
  // 排序
  sortComparer: (a, b) => b.createdAt.localeCompare(a.createdAt),
})

// 创建初始状态
const initialState = commentsAdapter.getInitialState({
  loading: false,
  error: null as string | null,
})

const commentSlice = createSlice({
  name: 'comments',
  initialState,
  reducers: {
    // 使用 adapter 的 CRUD 方法
    commentAdded: commentsAdapter.addOne,
    commentUpdated: commentsAdapter.updateOne,
    commentRemoved: commentsAdapter.removeOne,
    commentsReceived: commentsAdapter.setAll,
    commentsAddedMany: commentsAdapter.addMany,
  },
})

// Selectors — adapter 自动生成
export const {
  selectAll: selectAllComments,
  selectById: selectCommentById,
  selectIds: selectCommentIds,
  selectTotal: selectCommentTotal,
} = commentsAdapter.getSelectors((state: RootState) => state.comments)

// 使用
function CommentsList({ postId }: { postId: string }) {
  const comments = useAppSelector((state) =>
    selectAllComments(state).filter(c => c.postId === postId)
  )
  const total = useAppSelector(selectCommentTotal)

  return (
    <div>
      <h3>Comments ({total})</h3>
      {comments.map(comment => (
        <div key={comment.id}>{comment.text}</div>
      ))}
    </div>
  )
}
```

---

## 7. Selectors 与 Reselect

### 基础 Selector

```tsx
// 简单 selector
const selectTodos = (state: RootState) => state.todos.items
const selectFilter = (state: RootState) => state.todos.filter

// 使用 createSelector 创建 memoized selector
import { createSelector } from '@reduxjs/toolkit'

const selectFilteredTodos = createSelector(
  [selectTodos, selectFilter],
  (todos, filter) => {
    switch (filter) {
      case 'active': return todos.filter(t => !t.completed)
      case 'completed': return todos.filter(t => t.completed)
      default: return todos
    }
  }
)

const selectTodoStats = createSelector(
  [selectTodos],
  (todos) => ({
    total: todos.length,
    completed: todos.filter(t => t.completed).length,
    active: todos.filter(t => !t.completed).length,
    percentComplete: todos.length
      ? Math.round((todos.filter(t => t.completed).length / todos.length) * 100)
      : 0,
  })
)
```

### 参数化 Selector

```tsx
const selectTodoById = createSelector(
  [
    (state: RootState) => state.todos.items,
    (state: RootState, id: string) => id,
  ],
  (todos, id) => todos.find(t => t.id === id)
)

// 使用
function TodoItem({ id }: { id: string }) {
  const todo = useAppSelector((state) => selectTodoById(state, id))
  // ...
}
```

---

## 8. Middleware 自定义

```tsx
import { Middleware } from '@reduxjs/toolkit'

// 日志中间件
const loggerMiddleware: Middleware = (store) => (next) => (action) => {
  console.group(action.type)
  console.info('dispatching', action)
  const result = next(action)
  console.log('next state', store.getState())
  console.groupEnd()
  return result
}

// 配置
const store = configureStore({
  reducer: rootReducer,
  middleware: (getDefaultMiddleware) =>
    getDefaultMiddleware()
      .concat(loggerMiddleware)
      .concat(api.middleware),
})
```

---

## 总结

- `createSlice` 使用 Immer 直接修改 state，支持 `prepare` 回调
- `extraReducers` 响应其他 slice 的 action 或 thunk 的状态
- `createAsyncThunk` 处理异步操作，自动 dispatch pending/fulfilled/rejected
- **RTK Query** 是内建的数据获取方案，支持缓存、标签失效、轮询等
- `createEntityAdapter` 规范化列表数据，自动生成 CRUD 方法和 selector
- `createSelector`（reselect）创建 memoized selector 避免不必要的计算
