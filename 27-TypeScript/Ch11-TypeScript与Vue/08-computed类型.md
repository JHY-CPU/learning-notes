# computed类型

## 一、概念说明

`computed` 创建计算属性，TypeScript 自动推断其返回值类型为 `ComputedRef<T>`。计算属性的类型由 getter 函数的返回值决定，提供只读的响应式数据。

## 二、具体用法

### 2.1 基本类型推断

```typescript
import { ref, computed } from 'vue';

const firstName = ref('张');
const lastName = ref('三');

// 自动推断为 ComputedRef<string>
const fullName = computed(() => `${firstName.value}${lastName.value}`);
// fullName.value 类型是 string

// 数字计算
const count = ref(10);
const doubled = computed(() => count.value * 2);  // ComputedRef<number>

// 布尔计算
const items = ref<string[]>([]);
const hasItems = computed(() => items.value.length > 0); // ComputedRef<boolean>
```

### 2.2 复杂计算类型

```typescript
interface User {
  id: number;
  name: string;
  role: 'admin' | 'user';
  active: boolean;
}

const users = ref<User[]>([]);

// 返回值类型自动推断为 ComputedRef<User[]>
const activeUsers = computed(() => users.value.filter(u => u.active));

// 返回值类型自动推断为 ComputedRef<User | undefined>
const adminUser = computed(() => users.value.find(u => u.role === 'admin'));

// 计算派生数据
const userStats = computed(() => ({
  total: users.value.length,
  active: users.value.filter(u => u.active).length,
  admins: users.value.filter(u => u.role === 'admin').length,
}));
// 类型: ComputedRef<{ total: number; active: number; admins: number }>
```

### 2.3 可写计算属性

```typescript
const firstName = ref('张');
const lastName = ref('三');

// 可写的 computed
const fullName = computed({
  get() {
    return `${firstName.value} ${lastName.value}`;
  },
  set(newValue: string) {
    const [first, ...rest] = newValue.split(' ');
    firstName.value = first;
    lastName.value = rest.join(' ');
  },
});

fullName.value = '李 四'; // setter 被调用
// firstName.value === '李', lastName.value === '四'
```

### 2.4 在模板中使用

```vue
<script setup lang="ts">
import { ref, computed } from 'vue';

const price = ref(100);
const quantity = ref(2);
const discount = ref(0.1);

const subtotal = computed(() => price.value * quantity.value);
const total = computed(() => subtotal.value * (1 - discount.value));

// 格式化输出
const formattedTotal = computed(() => `¥${total.value.toFixed(2)}`);
</script>

<template>
  <p>小计: ¥{{ subtotal }}</p>
  <p>总价: {{ formattedTotal }}</p>
</template>
```

### 2.5 computed 与 watch 的选择

```typescript
// computed — 有返回值，同步计算
const filtered = computed(() => items.value.filter(item => item.active));

// watch — 无返回值，副作用操作
watch(count, (newVal) => {
  console.log('count 变化了:', newVal);
  localStorage.setItem('count', String(newVal));
});

// 规则：需要返回值用 computed，副作用用 watch
```

## 三、注意事项与常见陷阱

1. **computed 是惰性的**：只有依赖变化时才重新计算
2. **computed 是只读的**（默认）：修改 `computed.value` 会报警告
3. **getter 中不要有副作用**：computed 应只用于计算值
4. **computed 依赖 ref/reactive**：确保计算所用的数据是响应式的
5. **可写 computed 的 setter 参数类型与 getter 返回值一致**
