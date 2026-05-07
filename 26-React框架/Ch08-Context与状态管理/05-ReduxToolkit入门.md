# Redux Toolkit 入门

Redux Toolkit (RTK) 是 Redux 官方推荐的编写 Redux 逻辑的方式。它简化了 Redux 的配置和使用，内置了最佳实践。

---

## 一、安装与配置

### 1.1 安装

```bash
npm install @reduxjs/toolkit react-redux
```

### 1.2 创建 Store

```jsx
// store/index.js
import { configureStore } from '@reduxjs/toolkit';
import counterReducer from './counterSlice';
import todosReducer from './todosSlice';

const store = configureStore({
  reducer: {
    counter: counterReducer,
    todos: todosReducer,
  },
  // configureStore 自动配置：
  // - Redux DevTools Extension
  // - redux-thunk 中间件
  // - 开发环境的序列化检查中间件
});

export default store;
```

### 1.3 提供 Store

```jsx
// main.jsx / index.jsx
import { Provider } from 'react-redux';
import store from './store';

ReactDOM.createRoot(document.getElementById('root')).render(
  <Provider store={store}>
    <App />
  </Provider>
);
```

---

## 二、createSlice

`createSlice` 是 RTK 的核心 API，自动生成 action creators 和 action types。

### 2.1 基础 Slice

```jsx
// store/counterSlice.js
import { createSlice } from '@reduxjs/toolkit';

const counterSlice = createSlice({
  name: 'counter',       // slice 名称，用于生成 action type
  initialState: {
    value: 0,
  },
  reducers: {
    // 每个 reducer 函数对应一个 action
    // RTK 内置了 immer，可以直接"修改"状态
    increment: (state) => {
      state.value += 1;  // 看起来像直接修改，实际由 immer 处理为不可变更新
    },
    decrement: (state) => {
      state.value -= 1;
    },
    incrementByAmount: (state, action) => {
      state.value += action.payload;
    },
    reset: (state) => {
      state.value = 0;
    },
  },
});

// 自动生成 action creators
// counterSlice.actions.increment() → { type: 'counter/increment' }
// counterSlice.actions.incrementByAmount(5) → { type: 'counter/incrementByAmount', payload: 5 }
export const { increment, decrement, incrementByAmount, reset } = counterSlice.actions;

// 自动生成的 reducer
export default counterSlice.reducer;
```

### 2.2 Immer 的魔力

```jsx
// 传统 Redux（手动不可变更新）：
function todosReducer(state = [], action) {
  switch (action.type) {
    case 'ADD_TODO':
      return [...state, action.payload];  // 展开创建新数组
    case 'TOGGLE_TODO':
      return state.map(t =>
        t.id === action.payload ? { ...t, done: !t.done } : t
      );
    case 'REMOVE_TODO':
      return state.filter(t => t.id !== action.payload);
  }
}

// RTK（直接"修改"，immer 处理不可变性）：
const todosSlice = createSlice({
  name: 'todos',
  initialState: [],
  reducers: {
    addTodo: (state, action) => {
      state.push(action.payload);  // 直接 push！
    },
    toggleTodo: (state, action) => {
      const todo = state.find(t => t.id === action.payload);
      if (todo) todo.done = !todo.done;  // 直接修改！
    },
    removeTodo: (state, action) => {
      const index = state.findIndex(t => t.id === action.payload);
      if (index !== -1) state.splice(index, 1);  // 直接 splice！
    },
  },
});
```

### 2.3 嵌套状态更新

```jsx
const userSlice = createSlice({
  name: 'user',
  initialState: {
    profile: { name: '', email: '' },
    preferences: { theme: 'light', lang: 'zh' },
  },
  reducers: {
    updateProfile: (state, action) => {
      Object.assign(state.profile, action.payload);
    },
    setTheme: (state, action) => {
      state.preferences.theme = action.payload;
    },
  },
});
```

---

## 三、useSelector 和 useDispatch

### 3.1 useSelector — 读取状态

```jsx
import { useSelector } from 'react-redux';

function Counter() {
  // useSelector 接受 selector 函数，返回 store 中的部分状态
  const count = useSelector((state) => state.counter.value);

  return <span>{count}</span>;
}

// 也可以进行计算
function CompletedTodos() {
  const completedCount = useSelector((state) =>
    state.todos.filter(t => t.done).length
  );

  return <span>已完成: {completedCount}</span>;
}
```

### 3.2 useDispatch — 派发 Action

```jsx
import { useDispatch } from 'react-redux';
import { increment, decrement, incrementByAmount } from './counterSlice';

function Counter() {
  const count = useSelector((state) => state.counter.value);
  const dispatch = useDispatch();

  return (
    <div>
      <span>{count}</span>
      <button onClick={() => dispatch(decrement())}>-1</button>
      <button onClick={() => dispatch(increment())}>+1</button>
      <button onClick={() => dispatch(incrementByAmount(5))}>+5</button>
    </div>
  );
}
```

### 3.3 在子组件中使用

```jsx
// 组件不需要通过 props 接收状态
function TodoList() {
  const todos = useSelector((state) => state.todos);
  const dispatch = useDispatch();

  return (
    <ul>
      {todos.map(todo => (
        <TodoItem key={todo.id} todo={todo} />
      ))}
    </ul>
  );
}

function TodoItem({ todo }) {
  const dispatch = useDispatch();

  return (
    <li>
      <input
        type="checkbox"
        checked={todo.done}
        onChange={() => dispatch(toggleTodo(todo.id))}
      />
      {todo.text}
      <button onClick={() => dispatch(removeTodo(todo.id))}>删除</button>
    </li>
  );
}
```

---

## 四、createAsyncThunk — 异步操作

### 4.1 基础用法

```jsx
// store/usersSlice.js
import { createSlice, createAsyncThunk } from '@reduxjs/toolkit';

// 创建异步 thunk
export const fetchUsers = createAsyncThunk(
  'users/fetchUsers',           // action type 前缀
  async (_, { rejectWithValue }) => {
    try {
      const response = await fetch('/api/users');
      if (!response.ok) throw new Error('请求失败');
      return await response.json();  // 返回值作为 fulfilled action 的 payload
    } catch (error) {
      return rejectWithValue(error.message);  // 作为 rejected action 的 payload
    }
  }
);

const usersSlice = createSlice({
  name: 'users',
  initialState: {
    items: [],
    loading: false,
    error: null,
  },
  reducers: {
    // 同步 reducers
  },
  extraReducers: (builder) => {
    builder
      .addCase(fetchUsers.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(fetchUsers.fulfilled, (state, action) => {
        state.loading = false;
        state.items = action.payload;
      })
      .addCase(fetchUsers.rejected, (state, action) => {
        state.loading = false;
        state.error = action.payload;
      });
  },
});

export default usersSlice.reducer;
```

### 4.2 使用异步 Thunk

```jsx
import { useEffect } from 'react';
import { useSelector, useDispatch } from 'react-redux';
import { fetchUsers } from './usersSlice';

function UserList() {
  const dispatch = useDispatch();
  const { items: users, loading, error } = useSelector(state => state.users);

  useEffect(() => {
    dispatch(fetchUsers());
  }, [dispatch]);

  if (loading) return <Spinner />;
  if (error) return <div>错误: {error}</div>;

  return users.map(u => <div key={u.id}>{u.name}</div>);
}
```

### 4.3 带参数的异步 Thunk

```jsx
export const fetchUserById = createAsyncThunk(
  'users/fetchById',
  async (userId, { rejectWithValue }) => {
    const response = await fetch(`/api/users/${userId}`);
    if (!response.ok) return rejectWithValue('用户不存在');
    return await response.json();
  }
);

// 使用
dispatch(fetchUserById(123));
```

---

## 五、RTK Query — 数据请求

RTK Query 是 RTK 内置的数据请求和缓存解决方案，替代手动编写 createAsyncThunk。

### 5.1 定义 API

```jsx
// store/api.js
import { createApi, fetchBaseQuery } from '@reduxjs/toolkit/query/react';

export const api = createApi({
  reducerPath: 'api',
  baseQuery: fetchBaseQuery({ baseUrl: '/api' }),
  tagTypes: ['User', 'Post'],  // 用于缓存失效
  endpoints: (builder) => ({
    // 查询：GET 请求
    getUsers: builder.query({
      query: () => '/users',
      providesTags: ['User'],  // 此查询提供 'User' 标签
    }),

    getUserById: builder.query({
      query: (id) => `/users/${id}`,
      providesTags: (result, error, id) => [{ type: 'User', id }],
    }),

    // 变更：POST/PUT/DELETE 请求
    createUser: builder.mutation({
      query: (newUser) => ({
        url: '/users',
        method: 'POST',
        body: newUser,
      }),
      invalidatesTags: ['User'],  // 创建后使 'User' 缓存失效，自动重新获取
    }),

    updateUser: builder.mutation({
      query: ({ id, ...data }) => ({
        url: `/users/${id}`,
        method: 'PUT',
        body: data,
      }),
      invalidatesTags: (result, error, { id }) => [{ type: 'User', id }],
    }),

    deleteUser: builder.mutation({
      query: (id) => ({
        url: `/users/${id}`,
        method: 'DELETE',
      }),
      invalidatesTags: ['User'],
    }),
  }),
});

// 自动生成 hooks
export const {
  useGetUsersQuery,
  useGetUserByIdQuery,
  useCreateUserMutation,
  useUpdateUserMutation,
  useDeleteUserMutation,
} = api;
```

### 5.2 注册到 Store

```jsx
// store/index.js
import { api } from './api';

const store = configureStore({
  reducer: {
    [api.reducerPath]: api.reducer,
    // 其他 slices...
  },
  middleware: (getDefaultMiddleware) =>
    getDefaultMiddleware().concat(api.middleware),
});
```

### 5.3 在组件中使用

```jsx
import { useGetUsersQuery, useCreateUserMutation } from './store/api';

function UserList() {
  // 自动生成的 hook：自动加载、缓存、重新验证
  const { data: users, isLoading, error, refetch } = useGetUsersQuery();
  const [createUser, { isLoading: isCreating }] = useCreateUserMutation();

  const handleCreate = async () => {
    await createUser({ name: '新用户', email: 'new@example.com' });
    // 自动重新获取用户列表（因为 invalidatesTags）
  };

  if (isLoading) return <Spinner />;
  if (error) return <div>错误: {error.message}</div>;

  return (
    <div>
      <button onClick={handleCreate} disabled={isCreating}>
        {isCreating ? '创建中...' : '添加用户'}
      </button>
      {users.map(u => <div key={u.id}>{u.name}</div>)}
    </div>
  );
}
```

---

## 六、DevTools

### 6.1 默认集成

`configureStore` 默认启用 Redux DevTools Extension。安装浏览器扩展后，自动可用。

### 6.2 DevTools 功能

- **时间旅行调试**：查看每个 action 前后的状态
- **Action 列表**：所有被 dispatch 的 action
- **State 树**：完整的状态结构
- **Diff 视图**：状态变化的差异

### 6.3 自定义中间件日志

```jsx
const store = configureStore({
  reducer: rootReducer,
  middleware: (getDefaultMiddleware) =>
    getDefaultMiddleware()
      .concat(api.middleware)
      .concat((store) => (next) => (action) => {
        console.log('dispatching', action);
        const result = next(action);
        console.log('next state', store.getState());
        return result;
      }),
});
```

---

## 七、项目结构建议

### 7.1 Feature Folder 结构

```
src/
├── store/
│   ├── index.js              # configureStore
│   └── api.js                # RTK Query API 定义
├── features/
│   ├── auth/
│   │   ├── authSlice.js      # createSlice
│   │   ├── AuthLogin.jsx     # 组件
│   │   └── AuthGuard.jsx     # 组件
│   ├── todos/
│   │   ├── todosSlice.js
│   │   ├── TodoList.jsx
│   │   ├── TodoItem.jsx
│   │   └── AddTodo.jsx
│   └── users/
│       ├── usersSlice.js
│       └── UserList.jsx
├── components/               # 共享组件
└── App.jsx
```

### 7.2 何时使用 Redux

- 多个不相关的组件需要访问相同状态
- 状态更新逻辑复杂
- 需要时间旅行调试
- 团队需要统一的状态管理模式
- 中大型应用

### 7.3 何时不用 Redux

- 只有少量组件需要共享状态：用 Context
- 主要是服务端数据：用 TanStack Query
- 状态很简单：用 useState
- 不想引入额外依赖：用 Zustand
