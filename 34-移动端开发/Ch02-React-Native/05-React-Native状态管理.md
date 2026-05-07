# React Native 状态管理

## 一、概念说明

状态管理是 React Native 应用的核心议题。随着应用复杂度增加，组件间的状态共享和数据流管理变得至关重要。常用方案包括 Context API、Redux、MobX 和 Zustand 等。

```javascript
// React Hooks 基础状态管理
import { useState, useReducer, createContext, useContext } from 'react';

// useState 简单状态
const Counter = () => {
  const [count, setCount] = useState(0);
  return (
    <View>
      <Text>计数: {count}</Text>
      <Button title="+1" onPress={() => setCount(c => c + 1)} />
    </View>
  );
};
```

## 二、状态管理方案

### 2.1 Context API

```javascript
// 创建 Context
const ThemeContext = createContext();

// Provider 组件
const ThemeProvider = ({ children }) => {
  const [theme, setTheme] = useState('light');

  const toggleTheme = () => {
    setTheme(prev => prev === 'light' ? 'dark' : 'light');
  };

  return (
    <ThemeContext.Provider value={{ theme, toggleTheme }}>
      {children}
    </ThemeContext.Provider>
  );
};

// 消费 Context
const ThemedButton = () => {
  const { theme, toggleTheme } = useContext(ThemeContext);

  return (
    <TouchableOpacity
      style={{
        backgroundColor: theme === 'dark' ? '#333' : '#fff',
        padding: 16,
      }}
      onPress={toggleTheme}
    >
      <Text style={{ color: theme === 'dark' ? '#fff' : '#333' }}>
        切换主题
      </Text>
    </TouchableOpacity>
  );
};
```

### 2.2 useReducer 复杂状态

```javascript
// Reducer 定义
const todoReducer = (state, action) => {
  switch (action.type) {
    case 'ADD_TODO':
      return {
        ...state,
        todos: [...state.todos, {
          id: Date.now(),
          text: action.payload,
          completed: false,
        }],
      };
    case 'TOGGLE_TODO':
      return {
        ...state,
        todos: state.todos.map(todo =>
          todo.id === action.payload
            ? { ...todo, completed: !todo.completed }
            : todo
        ),
      };
    case 'DELETE_TODO':
      return {
        ...state,
        todos: state.todos.filter(todo => todo.id !== action.payload),
      };
    case 'SET_FILTER':
      return { ...state, filter: action.payload };
    default:
      return state;
  }
};

// 使用 useReducer
const TodoApp = () => {
  const [state, dispatch] = useReducer(todoReducer, {
    todos: [],
    filter: 'all',
  });

  const addTodo = (text) => {
    dispatch({ type: 'ADD_TODO', payload: text });
  };

  const toggleTodo = (id) => {
    dispatch({ type: 'TOGGLE_TODO', payload: id });
  };

  return (
    <View>
      <TextInput onSubmitEditing={(e) => addTodo(e.nativeEvent.text)} />
      {state.todos.map(todo => (
        <TouchableOpacity key={todo.id} onPress={() => toggleTodo(todo.id)}>
          <Text style={{ textDecorationLine: todo.completed ? 'line-through' : 'none' }}>
            {todo.text}
          </Text>
        </TouchableOpacity>
      ))}
    </View>
  );
};
```

### 2.3 Redux Toolkit

```javascript
// store/userSlice.js
import { createSlice, createAsyncThunk } from '@reduxjs/toolkit';

// 异步操作
export const fetchUser = createAsyncThunk(
  'user/fetchUser',
  async (userId, { rejectWithValue }) => {
    try {
      const response = await fetch(`/api/users/${userId}`);
      return await response.json();
    } catch (error) {
      return rejectWithValue(error.message);
    }
  }
);

const userSlice = createSlice({
  name: 'user',
  initialState: {
    data: null,
    loading: false,
    error: null,
  },
  reducers: {
    updateUser: (state, action) => {
      state.data = { ...state.data, ...action.payload };
    },
    logout: (state) => {
      state.data = null;
      state.error = null;
    },
  },
  extraReducers: (builder) => {
    builder
      .addCase(fetchUser.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(fetchUser.fulfilled, (state, action) => {
        state.loading = false;
        state.data = action.payload;
      })
      .addCase(fetchUser.rejected, (state, action) => {
        state.loading = false;
        state.error = action.payload;
      });
  },
});

export const { updateUser, logout } = userSlice.actions;
export default userSlice.reducer;

// store/store.js
import { configureStore } from '@reduxjs/toolkit';
import userReducer from './userSlice';

export const store = configureStore({
  reducer: {
    user: userReducer,
  },
});

// 使用
import { useSelector, useDispatch } from 'react-redux';

const ProfileScreen = () => {
  const dispatch = useDispatch();
  const { data: user, loading } = useSelector(state => state.user);

  useEffect(() => {
    dispatch(fetchUser(1));
  }, []);

  if (loading) return <ActivityIndicator />;

  return (
    <View>
      <Text>{user?.name}</Text>
      <Button title="退出" onPress={() => dispatch(logout())} />
    </View>
  );
};
```

### 2.4 Zustand（轻量级方案）

```javascript
// store/useStore.js
import { create } from 'zustand';

const useStore = create((set, get) => ({
  // 状态
  count: 0,
  user: null,
  cart: [],

  // 操作
  increment: () => set(state => ({ count: state.count + 1 })),
  decrement: () => set(state => ({ count: state.count - 1 })),
  setUser: (user) => set({ user }),
  addToCart: (item) => set(state => ({
    cart: [...state.cart, item],
  })),
  removeFromCart: (id) => set(state => ({
    cart: state.cart.filter(item => item.id !== id),
  })),
  getTotal: () => {
    const { cart } = get();
    return cart.reduce((sum, item) => sum + item.price * item.quantity, 0);
  },
}));

// 使用
const CartScreen = () => {
  const { cart, addToCart, removeFromCart, getTotal } = useStore();

  return (
    <View>
      {cart.map(item => (
        <View key={item.id}>
          <Text>{item.name} x{item.quantity}</Text>
          <TouchableOpacity onPress={() => removeFromCart(item.id)}>
            <Text>删除</Text>
          </TouchableOpacity>
        </View>
      ))}
      <Text>总计: ¥{getTotal()}</Text>
    </View>
  );
};
```

## 三、方案对比

| 方案 | 复杂度 | 包大小 | 学习曲线 | 适用场景 |
|------|--------|--------|----------|----------|
| Context + useReducer | 低 | 0 | 低 | 小型应用、局部状态 |
| Redux Toolkit | 中 | ~16kb | 中 | 大型应用、复杂状态 |
| Zustand | 低 | ~1kb | 低 | 中小型应用 |
| MobX | 中 | ~16kb | 中 | 响应式编程偏好者 |

## 四、注意事项与常见陷阱

1. **不要过度设计**：简单状态用 useState，不要为每个功能都引入状态管理库
2. **避免状态冗余**：派生数据应该通过计算得到，而非单独存储
3. **合理拆分 Store**：按功能模块拆分，避免单一巨大的 Store
4. **不可变更新**：始终创建新对象，不要直接修改状态
5. **性能优化**：使用选择器（Selector）订阅最小状态，避免不必要的重渲染
