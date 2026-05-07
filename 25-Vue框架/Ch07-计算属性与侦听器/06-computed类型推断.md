# computed 类型推断

## 一、概念说明
TypeScript 中，`computed` 会自动推断返回值类型。也可以显式指定泛型类型来获得更精确的类型提示。

## 二、具体用法

### 2.1 自动类型推断
```vue
<script setup lang="ts">
import { ref, computed } from 'vue'

const count = ref(0)
// 类型自动推断为 ComputedRef<number>
const doubled = computed(() => count.value * 2)

const name = ref('Vue')
// 类型自动推断为 ComputedRef<string>
const greeting = computed(() => `Hello, ${name.value}`)
</script>
```

### 2.2 显式指定类型
```vue
<script setup lang="ts">
import { ref, computed } from 'vue'

interface User {
  id: number
  name: string
  role: 'admin' | 'user'
}

const users = ref<User[]>([])

// 显式指定返回类型
const admins = computed<User[]>(() =>
  users.value.filter(u => u.role === 'admin')
)
</script>
```

### 2.3 可写 computed 类型
```vue
<script setup lang="ts">
import { ref, computed } from 'vue'

const celsius = ref(0)

const fahrenheit = computed({
  get: (): number => celsius.value * 9 / 5 + 32,
  set: (val: number) => {
    celsius.value = (val - 32) * 5 / 9
  }
})
</script>
```

### 2.4 联合类型 computed
```vue
<script setup lang="ts">
import { ref, computed } from 'vue'

const status = ref<'loading' | 'success' | 'error'>('loading')
const message = computed<string>(() => {
  switch (status.value) {
    case 'loading': return '加载中...'
    case 'success': return '加载成功'
    case 'error': return '加载失败'
  }
})
</script>
```

## 三、注意事项与常见陷阱
- computed 的泛型参数指定的是**返回值类型**，不是依赖类型
- 如果依赖的数据类型正确，大多数情况下不需要显式指定类型
- 可写 computed 的 getter 和 setter 类型应一致
- 使用 `ComputedRef<T>` 类型在函数间传递 computed
