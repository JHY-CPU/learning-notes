# watch类型

## 一、概念说明

`watch` 和 `watchEffect` 是 Vue 3 中的副作用 API。TypeScript 可以自动推断监听源的类型和回调函数的参数类型，确保回调中能安全地访问新旧值。

## 二、具体用法

### 2.1 watch 基本类型

```typescript
import { ref, watch } from 'vue';

const count = ref(0);

// TypeScript 自动推断回调参数类型
watch(count, (newValue, oldValue) => {
  // newValue: number
  // oldValue: number
  console.log(`从 ${oldValue} 变为 ${newValue}`);
});

// 监听 reactive 对象的属性
const state = reactive({ name: '张三', age: 25 });

// 使用 getter 函数
watch(
  () => state.age,
  (newAge, oldAge) => {
    // newAge: number
    // oldAge: number
  }
);
```

### 2.2 监听多个源

```typescript
const firstName = ref('张');
const lastName = ref('三');

// 监听多个 ref — 参数是数组
watch([firstName, lastName], ([newFirst, newLast], [oldFirst, oldLast]) => {
  console.log(`${newFirst}${newLast} ← ${oldFirst}${oldLast}`);
});

// 监听 reactive 的多个属性
watch(
  [() => state.name, () => state.age],
  ([newName, newAge], [oldName, oldAge]) => {
    // 自动推断类型
  }
);
```

### 2.3 深度监听

```typescript
interface User {
  profile: {
    name: string;
    address: {
      city: string;
    };
  };
}

const user = ref<User>({
  profile: { name: '张三', address: { city: '北京' } },
});

// 深度监听 — 对象内部任何属性变化都触发
watch(
  user,
  (newUser) => {
    // newUser: User
    console.log(newUser.profile.address.city);
  },
  { deep: true }
);

// 监听 reactive 对象默认就是深度的
const state = reactive({ user: { name: '张三' } });
watch(state, (newState) => {
  console.log(newState.user.name);
});
```

### 2.4 watchEffect 类型

```typescript
import { ref, watchEffect } from 'vue';

const query = ref('');
const page = ref(1);

// watchEffect 自动追踪依赖
watchEffect(() => {
  // 自动追踪 query.value 和 page.value
  console.log(`搜索: ${query.value}, 页码: ${page.value}`);
  // 参数类型自动推断
});

// 异步 watchEffect
watchEffect(async () => {
  const data = await fetch(`/api/search?q=${query.value}&page=${page.value}`);
  // ...
});
```

### 2.5 WatchSource 类型

```typescript
import type { WatchSource, WatchCallback } from 'vue';

// WatchSource — watch 的第一个参数类型
// 可以是 Ref、getter 函数、或它们的数组
type MyWatchSource<T> = Ref<T> | (() => T);

// 自定义 watch 包装
function useWatch<T>(
  source: WatchSource<T>,
  callback: WatchCallback<T, T | undefined>,
  options?: { immediate?: boolean; deep?: boolean }
) {
  watch(source, callback, options);
}
```

### 2.6 onWatcherCleanup（Vue 3.5+）

```typescript
import { ref, watch, onWatcherCleanup } from 'vue';

const userId = ref(1);

watch(userId, (id) => {
  const controller = new AbortController();

  // 清理上一次的请求
  onWatcherCleanup(() => {
    controller.abort();
  });

  fetch(`/api/users/${id}`, { signal: controller.signal });
});
```

## 三、注意事项与常见陷阱

1. **`watch` 的回调参数类型与监听源一致**：`Ref<number>` 的新旧值都是 `number`
2. **监听 reactive 对象时新旧值相同**：都是同一个代理对象的引用
3. **`watchEffect` 不需要指定监听源**：自动追踪回调中用到的所有响应式数据
4. **深度监听有性能开销**：只在必要时使用 `deep: true`
5. **异步回调中的 `watchEffect`**：不能正确追踪异步后的依赖
