# defineEmits泛型

## 一、概念说明

`defineEmits` 定义组件可以触发的事件及其参数类型。泛型语法让事件的参数有精确的类型约束，确保父组件接收到正确的事件数据。

## 二、具体用法

### 2.1 基本泛型事件

```vue
<script setup lang="ts">
// 泛型语法定义事件
const emit = defineEmits<{
  change: [value: string];           // 带命名参数
  submit: [data: { name: string }];  // 对象参数
  close: [];                         // 无参数
  'update:modelValue': [value: number]; // v-model 事件
}>();

// 使用 — 类型安全
emit('change', '新值');
emit('submit', { name: '张三' });
emit('close');
emit('update:modelValue', 42);

// 错误 — 编译报错
// emit('change', 123);           // 类型错误：不能将 number 赋给 string
// emit('submit', { wrong: 1 });  // 类型错误：缺少 name 属性
</script>
```

### 2.2 使用接口定义

```vue
<script setup lang="ts">
// 用 interface 定义事件映射
interface Events {
  select: [item: Item];
  delete: [id: number];
  reorder: [from: number, to: number];
  error: [error: Error];
}

const emit = defineEmits<Events>();
</script>
```

### 2.3 表单事件

```vue
<script setup lang="ts">
const emit = defineEmits<{
  'update:modelValue': [value: string];
  blur: [event: FocusEvent];
  focus: [event: FocusEvent];
  validate: [isValid: boolean, errors: string[]];
}>();

const handleInput = (e: Event) => {
  const value = (e.target as HTMLInputElement).value;
  emit('update:modelValue', value);
};
</script>
```

### 2.4 配合 defineProps 实现 v-model

```vue
<!-- 子组件 -->
<script setup lang="ts">
const props = defineProps<{
  modelValue: string;
}>();

const emit = defineEmits<{
  'update:modelValue': [value: string];
}>();

const updateValue = (value: string) => {
  emit('update:modelValue', value);
};
</script>

<!-- 多个 v-model -->
<script setup lang="ts">
defineProps<{
  modelValue: string;
  title: string;
}>();

defineEmits<{
  'update:modelValue': [value: string];
  'update:title': [value: string];
}>();
</script>

<!-- 父组件使用 -->
<!-- <MyComponent v-model="name" v-model:title="titleText" /> -->
```

### 2.5 运行时声明（不推荐）

```vue
<script setup lang="ts">
// 运行时声明 — 不如泛型语法简洁
const emit = defineEmits({
  change: (value: string) => typeof value === 'string',
  submit: null, // 不做验证
});
</script>
```

## 三、注意事项与常见陷阱

1. **事件参数用元组语法**：`[value: string]` 而非 `(value: string) => void`
2. **命名参数是可选的**：`[string]` 和 `[value: string]` 在类型上等价
3. **`v-model` 事件名是 `'update:modelValue'`**：注意引号
4. **多 v-model 用 `'update:propName'` 格式**：与 Vue 3 多 v-model 对应
5. **不要混用泛型语法和运行时声明**：选择其中一种
