# ref与reactive类型

## 一、概念说明

`ref` 和 `reactive` 是 Vue 3 响应式系统的核心。TypeScript 对它们的类型推断非常精确：`ref` 返回 `Ref<T>`，`reactive` 返回代理对象的类型。理解它们的类型差异有助于正确选择使用哪种响应式 API。

## 二、具体用法

### 2.1 ref 类型

```typescript
import { ref } from 'vue';

// 基本类型 — 自动推断
const count = ref(0);           // Ref<number>
const name = ref('');           // Ref<string>
const active = ref(false);      // Ref<boolean>
const items = ref<string[]>([]); // Ref<string[]> — 需要泛型

// 访问值 — .value
count.value++;                  // OK
name.value = '张三';            // OK

// 泛型 ref
const nullable = ref<User | null>(null); // Ref<User | null>
```

### 2.2 reactive 类型

```typescript
import { reactive } from 'vue';

// 自动推断类型
const state = reactive({
  count: 0,
  name: '',
  items: [] as string[],
});

// state 的类型是 { count: number; name: string; items: string[] }
// 不需要 .value
state.count++;
state.name = '张三';

// 接口定义
interface FormState {
  username: string;
  email: string;
  age: number;
  agreed: boolean;
}

const form = reactive<FormState>({
  username: '',
  email: '',
  age: 0,
  agreed: false,
});
```

### 2.3 UnwrapRef 类型

```typescript
import { reactive, ref, type UnwrapRef } from 'vue';

// reactive 自动解包内部的 ref
const count = ref(0);
const state = reactive({ count });

// state.count 类型是 number，不是 Ref<number>
console.log(state.count); // 0 — 自动解包

// UnwrapRef — 解包后的类型
type RawState = { count: Ref<number> };
type UnwrappedState = UnwrapRef<RawState>; // { count: number }
```

### 2.4 readonly 与类型

```typescript
import { ref, reactive, readonly } from 'vue';

// readonly 创建只读版本
const original = ref(0);
const readOnly = readonly(original);
// readOnly.value = 1; // 错误：只读

const state = reactive({ count: 0 });
const readOnlyState = readonly(state);
// readOnlyState.count = 1; // 错误：只读

// 类型变化
// Ref<number> → Readonly<Ref<number>>
// { count: number } → Readonly<{ count: number }>
```

### 2.5 ref 与 reactive 的选择

```typescript
// ref — 适合基本类型和独立值
const count = ref(0);
const isOpen = ref(false);
const selectedId = ref<number | null>(null);

// reactive — 适合对象状态
const formState = reactive({
  username: '',
  password: '',
  remember: false,
});

// 不推荐 — 对 ref 使用 reactive
// const state = reactive({ count: ref(0) }); // 冗余

// 推荐 — 简单场景用 ref
const user = ref<{ name: string; age: number } | null>(null);

// 推荐 — 复杂表单用 reactive
const form = reactive({
  personal: { name: '', email: '' },
  address: { street: '', city: '' },
  preferences: { newsletter: false },
});
```

### 2.6 toRefs 类型

```typescript
import { reactive, toRefs } from 'vue';

const state = reactive({
  count: 0,
  name: '',
});

// toRefs 将 reactive 对象转为 ref 对象
const { count, name } = toRefs(state);
// count: Ref<number>
// name: Ref<string>

// 可以安全地解构和传递
count.value++;
```

## 三、注意事项与常见陷阱

1. **`ref` 在模板中自动解包**：不需要 `.value`
2. **`ref` 在 `reactive` 中自动解包**：访问时不需要 `.value`
3. **`reactive` 不能重新赋值**：会丢失响应性
4. **基本类型必须用 `ref`**：`reactive` 不适用于基本类型
5. **`toRefs` 保持响应性**：解构 `reactive` 对象时使用
