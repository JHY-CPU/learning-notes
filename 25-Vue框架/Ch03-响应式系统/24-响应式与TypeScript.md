# 响应式与 TypeScript

## 一、概念说明

Vue 3 完全用 TypeScript 重写，提供了优秀的类型支持。`ref()`、`reactive()`、`computed()` 等 API 都有完善的泛型定义。合理使用类型标注可以提升开发体验和代码质量。

```vue
<script setup lang="ts">
import { ref, reactive, computed, type Ref } from 'vue'

// ref 类型推断
const count = ref(0)        // Ref<number>
const name = ref('Vue')     // Ref<string>

// 显式类型标注
const id = ref<number | null>(null)
const items = ref<string[]>([])

// reactive 类型推断
const state = reactive({
  count: 0,
  name: 'Vue'
}) // { count: number; name: string }

// ComputedRef 类型
const doubled = computed(() => count.value * 2) // ComputedRef<number>
</script>
```

## 二、具体用法

### 2.1 Ref 类型

```ts
import { ref, type Ref } from 'vue'

// 自动推断
const count = ref(0)           // Ref<number>
const message = ref('hello')   // Ref<string>

// 显式标注
const count2 = ref<number>(0)
const nullable = ref<string | null>(null)

// Ref 类型别名
const myRef: Ref<number> = ref(42)
```

### 2.2 Reactive 类型

```ts
import { reactive, type UnwrapNestedRefs } from 'vue'

interface User {
  name: string
  age: number
}

// 显式标注 reactive
const user = reactive<User>({
  name: '张三',
  age: 25
})

// reactive 返回 UnwrapNestedRefs<T>
const state: UnwrapNestedRefs<{ count: number }> = reactive({ count: 0 })
```

### 2.3 Computed 类型

```ts
import { ref, computed, type ComputedRef } from 'vue'

const count = ref(0)

// 自动推断为 ComputedRef<number>
const doubled = computed(() => count.value * 2)

// 显式标注
const formatted = computed<string>(() => `计数: ${count.value}`)
```

### 2.4 类型工具

```ts
import { type ToRefs, type DeepReadonly } from 'vue'

// toRefs 返回 ToRefs<T>
// readonly 返回 DeepReadonly<T>
```

## 三、注意事项与常见陷阱

- ref 在模板中自动解包，类型从 Ref<T> 变为 T
- reactive 的解构需要 toRefs 保持类型正确
- `ref<T>()` 的泛型参数应该与初始值类型一致
- computed 的返回类型一般由返回值自动推断
- 使用 `MaybeRef<T>` 类型标注可能为 ref 或普通值的参数
