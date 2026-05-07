# TypeScript与Vue3概述

## 一、概念说明

Vue 3 从设计之初就对 TypeScript 提供了一等公民级别的支持。Composition API 天然适合类型推断，`defineProps`、`defineEmits` 等宏提供了完整的泛型支持。

**核心优势**：Vue 3 + TypeScript 的类型推断能力远超 Vue 2，几乎不需要额外的类型标注。

## 二、具体用法

### 2.1 Vue 3 + TS 特性一览

```typescript
// Vue 3 的 TypeScript 特性
// 1. defineProps 支持泛型
const props = defineProps<{
  title: string;
  count: number;
  items?: string[];
}>();

// 2. defineEmits 支持泛型
const emit = defineEmits<{
  change: [id: number];
  submit: [data: FormData];
}>();

// 3. ref/reactive 自动推断类型
const count = ref(0);        // Ref<number>
const user = reactive({       // { name: string; age: number }
  name: '张三',
  age: 25,
});

// 4. computed 自动推断返回类型
const doubled = computed(() => count.value * 2); // ComputedRef<number>
```

### 2.2 对比 Vue 2 + TS

```typescript
// Vue 2 + TS — 需要装饰器（已不推荐）
// @Component
// export default class MyComponent extends Vue {
//   @Prop({ type: String }) readonly title!: string;
//   @Emit('change') onChange(id: number) { return id; }
// }

// Vue 3 + TS — 简洁自然
const props = defineProps<{ title: string }>();
const emit = defineEmits<{ change: [id: number] }>();
```

### 2.3 项目技术栈

| 方案 | 说明 |
|------|------|
| Vite + Vue 3 + TS | 推荐方案，快速开发 |
| Nuxt 3 + TS | 全栈/SSR 项目 |
| Vue CLI + TS | 旧项目迁移 |

### 2.4 核心类型工具

```typescript
import type {
  Ref,               // ref() 的返回类型
  ComputedRef,       // computed() 的返回类型
  UnwrapRef,         // reactive 解包后的类型
  ToRefs,            // toRefs() 的返回类型
  PropType,          // 运行时 props 声明的类型工具
  Component,         // 组件类型
  DefineComponent,   // defineComponent 的返回类型
  Slots,             // 插槽类型
  Directive,         // 自定义指令类型
} from 'vue';
```

## 三、注意事项与常见陷阱

1. **Composition API 优于 Options API**：类型推断更好
2. **使用 `<script setup lang="ts">`**：最简洁的 TS 写法
3. **不需要手动标注大多数类型**：Vue 能自动推断
4. **`defineProps` 的泛型语法是编译时宏**：不是普通的 TypeScript
5. **确保 `vue-tsc` 版本与 Vue 版本匹配**：避免类型检查不一致
