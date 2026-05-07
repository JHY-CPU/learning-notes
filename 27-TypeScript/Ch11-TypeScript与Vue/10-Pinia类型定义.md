# Pinia类型定义

## 一、概念说明

Pinia 是 Vue 3 的官方状态管理库，对 TypeScript 提供了一流支持。`defineStore` 自动推断 state、getters 和 actions 的类型，无需手动标注。

## 二、具体用法

### 2.1 Setup Store（推荐）

```typescript
import { defineStore } from 'pinia';
import { ref, computed } from 'vue';

// Setup 语法 — 类型推断最好
export const useCounterStore = defineStore('counter', () => {
  const count = ref(0);
  const name = ref('计数器');

  const doubleCount = computed(() => count.value * 2);

  const increment = () => count.value++;
  const decrement = () => count.value--;
  const reset = () => (count.value = 0);

  return { count, name, doubleCount, increment, decrement, reset };
});

// 使用时类型完整推断
const store = useCounterStore();
store.count;        // number
store.doubleCount;  // number
store.increment();  // () => void
```

### 2.2 Options Store

```typescript
interface UserState {
  users: User[];
  currentUser: User | null;
  loading: boolean;
}

export const useUserStore = defineStore('user', {
  state: (): UserState => ({
    users: [],
    currentUser: null,
    loading: false,
  }),

  getters: {
    // 自动推断返回类型
    activeUsers: (state) => state.users.filter(u => u.active),

    // 带参数的 getter（返回函数）
    getUserById: (state) => {
      return (id: number) => state.users.find(u => u.id === id);
    },

    // 使用其他 getter
    activeUserCount(): number {
      return this.activeUsers.length;
    },
  },

  actions: {
    async fetchUsers() {
      this.loading = true;
      try {
        this.users = await api.getUsers();
      } finally {
        this.loading = false;
      }
    },

    setCurrentUser(user: User | null) {
      this.currentUser = user;
    },
  },
});
```

### 2.3 Store 类型提取

```typescript
import type { Store, StoreDefinition } from 'pinia';

// 提取 store 的 state 类型
const store = useUserStore();
type UserStore = typeof store;
// 类型包含所有 state、getters 和 actions

// 提取特定类型
type UserState = ReturnType<typeof useUserStore>['$state'];

// Store 实例类型
type CounterStore = Store<'counter', CounterState, CounterGetters, CounterActions>;
```

### 2.4 Store 之间互相使用

```typescript
export const useCartStore = defineStore('cart', () => {
  const items = ref<CartItem[]>([]);

  const totalPrice = computed(() =>
    items.value.reduce((sum, item) => sum + item.price * item.quantity, 0)
  );

  // 在 store 中使用其他 store
  const userStore = useUserStore();

  const checkout = async () => {
    if (!userStore.currentUser) {
      throw new Error('请先登录');
    }
    // 结账逻辑
  };

  return { items, totalPrice, checkout };
});
```

### 2.5 插件类型

```typescript
import type { PiniaPluginContext } from 'pinia';

// Pinia 插件
function resetPlugin({ store }: PiniaPluginContext) {
  // 为每个 store 添加 $reset 方法
  const initialState = JSON.parse(JSON.stringify(store.$state));

  store.$reset = () => {
    store.$patch(initialState);
  };
}

// 类型扩展
declare module 'pinia' {
  export interface PiniaCustomProperties {
    $reset: () => void;
  }
}
```

## 三、注意事项与常见陷阱

1. **Setup Store 类型推断最好**：推荐使用 Composition API 语法
2. **Options Store 的 `state` 需要返回函数**：确保每个 store 实例有独立的状态
3. **不要在 store 中直接修改其他 store 的 state**：通过 actions 操作
4. **`storeToRefs` 保持响应性**：解构 store 时使用
5. **Store 的类型导出**：`export type { ... }` 而非直接 `export`
