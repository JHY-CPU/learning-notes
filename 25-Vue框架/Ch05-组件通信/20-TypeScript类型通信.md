# TypeScript 类型通信

## 一、概念说明
TypeScript 可以为 props、emit、provide/inject 等通信方式提供**编译时类型检查**，减少运行时错误，提升开发体验。

## 二、具体用法

### 2.1 类型化 Props
```vue
<script setup>
interface User {
  id: number
  name: string
  email: string
}

// 使用泛型定义 props
const props = withDefaults(defineProps<{
  user: User
  size?: 'sm' | 'md' | 'lg'
  active?: boolean
}>(), {
  size: 'md',
  active: false
})
</script>
```

### 2.2 类型化 Emit
```vue
<script setup>
const emit = defineEmits<{
  change: [value: string]
  submit: [data: { name: string; age: number }]
  close: []
}>()

// 编译时检查
emit('change', 'hello')  // ✓
emit('change', 123)      // ✗ 类型错误
</script>
```

### 2.3 类型化 Provide/Inject
```ts
// keys.ts
import type { InjectionKey, Ref } from 'vue'

export const THEME_KEY: InjectionKey<Ref<string>> = Symbol('theme')
export const USER_KEY: InjectionKey<User> = Symbol('user')
```

```vue
<!-- 祖先组件 -->
<script setup>
import { ref, provide } from 'vue'
import { THEME_KEY } from './keys'
provide(THEME_KEY, ref('dark'))
</script>

<!-- 后代组件 -->
<script setup>
import { inject } from 'vue'
import { THEME_KEY } from './keys'
const theme = inject(THEME_KEY)  // 类型: Ref<string> | undefined
</script>
```

### 2.4 泛型组件
```vue
<script setup lang="ts" generic="T">
const props = defineProps<{
  items: T[]
  onSelect: (item: T) => void
}>()
</script>
```

## 三、注意事项与常见陷阱
- inject 返回值可能为 `undefined`，需要处理或提供默认值
- `withDefaults` 只能与 `defineProps` 类型声明一起使用
- 泛型组件需要 `generic` 属性（Vue 3.3+）
- 建议在项目中统一管理 InjectionKey，避免散落各处
