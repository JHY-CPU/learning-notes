# Redux 详解

## 一、概念说明

Redux 是 JavaScript 应用的可预测状态容器，遵循单向数据流原则。Redux Toolkit 是官方推荐的简化写法。

```javascript
// Redux 三大原则
/*
1. 单一数据源 - 整个应用的状态存储在一个 Store 中
2. 状态只读 - 只能通过 dispatch action 来改变状态
3. 纯函数修改 - 使用纯函数（reducer）来处理状态变更
*/
```

## 二、Redux Toolkit 使用

```javascript
// store/counterSlice.js
import { createSlice } from '@reduxjs/toolkit';

const counterSlice = createSlice({
  name: 'counter',
  initialState: { value: 0 },
  reducers: {
    increment: (state) => { state.value += 1 },
    decrement: (state) => { state.value -= 1 },
    incrementByAmount: (state, action) => {
      state.value += action.payload;
    },
  },
});

export const { increment, decrement, incrementByAmount } = counterSlice.actions;
export default counterSlice.reducer;
```

```javascript
// store/store.js
import { configureStore } from '@reduxjs/toolkit';
import counterReducer from './counterSlice';
import userReducer from './userSlice';

export const store = configureStore({
  reducer: {
    counter: counterReducer,
    user: userReducer,
  },
});
```

```javascript
// 使用
import { useSelector, useDispatch } from 'react-redux';
import { increment, decrement } from './store/counterSlice';

const Counter = () => {
  const count = useSelector((state) => state.counter.value);
  const dispatch = useDispatch();

  return (
    <View>
      <Text>{count}</Text>
      <Button onPress={() => dispatch(increment())} title="+" />
      <Button onPress={() => dispatch(decrement())} title="-" />
    </View>
  );
};
```

## 三、异步操作

```javascript
// createAsyncThunk
import { createSlice, createAsyncThunk } from '@reduxjs/toolkit';

export const fetchUser = createAsyncThunk(
  'user/fetchUser',
  async (userId, { rejectWithValue }) => {
    try {
      const response = await fetch(`/api/users/${userId}`);
      if (!response.ok) throw new Error('请求失败');
      return await response.json();
    } catch (error) {
      return rejectWithValue(error.message);
    }
  }
);

const userSlice = createSlice({
  name: 'user',
  initialState: { data: null, loading: false, error: null },
  reducers: {},
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
```

## 四、性能优化

```javascript
// 使用 createSelector 创建记忆化 selector
import { createSelector } from '@reduxjs/toolkit';

const selectCartItems = (state) => state.cart.items;

export const selectCartTotal = createSelector(
  [selectCartItems],
  (items) => items.reduce((sum, item) => sum + item.price * item.quantity, 0)
);

// 使用 shallowEqual 避免不必要的重渲染
import { useSelector, shallowEqual } from 'react-redux';

const items = useSelector(
  (state) => state.cart.items,
  shallowEqual
);
```

## 五、注意事项

1. **不要在 reducer 中写副作用**：异步操作放在 createAsyncThunk 中
2. **合理拆分 slice**：按功能模块拆分，避免单一巨大 slice
3. **使用 immer**：Redux Toolkit 内置 immer，可直接修改 state
4. **中间件使用**：日志、异步等通过中间件处理
5. **DevTools**：开发时使用 Redux DevTools 追踪状态变化
