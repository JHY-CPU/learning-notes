# TypeScript与组合式API

## 一、概念说明

组合式API天然对TypeScript友好，`<script setup lang="ts">`提供完整的类型推断。泛型、接口、类型守卫都可以直接使用。

```vue
<script setup lang="ts">
import { ref, computed } from 'vue'

// ref自动推断类型 Ref<string>
const message = ref('hello')

// 显式指定类型
const count = ref<number>(0)
const items = ref<string[]>([])

// computed自动推断返回类型
const doubled = computed(() => count.value * 2)
</script>
```

## 二、具体用法

### defineProps类型

```vue
<script setup lang="ts">
// 运行时声明
const props = defineProps({
  title: { type: String, required: true },
  count: { type: Number, default: 0 }
})

// TypeScript类型声明（推荐）
interface Props {
  title: string
  count?: number
  items?: string[]
}

const props = withDefaults(defineProps<Props>(), {
  count: 0,
  items: () => []
})
</script>
```

### defineEmits类型

```vue
<script setup lang="ts">
// TS类型声明
const emit = defineEmits<{
  change: [id: number, name: string]
  update: [value: string]
  delete: [id: number]
}>()

emit('change', 1, '张三')
emit('update', '新值')
</script>
```

### 泛型组合式函数

```ts
// composables/useApi.ts
import { ref, Ref } from 'vue'

interface UseApiReturn<T> {
  data: Ref<T | null>
  loading: Ref<boolean>
  error: Ref<Error | null>
  execute: () => Promise<void>
}

export function useApi<T>(url: string): UseApiReturn<T> {
  const data = ref<T | null>(null)
  const loading = ref(false)
  const error = ref<Error | null>(null)

  const execute = async () => {
    loading.value = true
    try {
      const res = await fetch(url)
      data.value = await res.json()
    } catch (e) {
      error.value = e as Error
    } finally {
      loading.value = false
    }
  }

  return { data, loading, error, execute }
}
```

## 三、注意事项与常见陷阱

1. `ref<T>()`的泛型参数是内部值类型，不是ref类型本身
2. `reactive()`会自动解包内部ref，类型需注意
3. `defineProps`和`defineEmits`只能用`<script setup lang="ts">`中的语法
4. 为`provide/inject`添加类型建议使用Symbol作key并附带类型
5. 组合式函数的返回类型建议显式声明，增强IDE提示
