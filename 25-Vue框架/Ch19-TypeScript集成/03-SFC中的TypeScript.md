# SFC中的TypeScript

## 一、概念说明

Vue SFC（单文件组件）通过 `<script lang="ts">` 启用 TypeScript。`<script setup>` 与 TypeScript 结合时，Vue 编译器会自动处理类型推断，Props、Emits、Slots 等都支持完整的类型定义。

## 二、具体用法

### 基本 SFC + TS

```vue
<script setup lang="ts">
// 普通变量自动类型推断
const message = '你好 TypeScript'
// message 类型推断为 string

const count = ref(0)
// count 类型为 Ref<number>

const user = reactive({
  name: '张三',
  age: 25,
  hobbies: ['编程', '阅读']
})
// user 类型自动推断为 { name: string; age: number; hobbies: string[] }
</script>

<template>
  <div>
    <h1>{{ message }}</h1>
    <p>{{ user.name }}, {{ user.age }}岁</p>
    <p>计数: {{ count }}</p>
    <button @click="count++">增加</button>
  </div>
</template>
```

### 接口定义与使用

```vue
<script setup lang="ts">
// 定义数据接口
interface Product {
  id: number
  name: string
  price: number
  inStock: boolean
  tags: string[]
}

const products = ref<Product[]>([
  { id: 1, name: 'Vue课程', price: 99, inStock: true, tags: ['前端', '框架'] },
  { id: 2, name: 'TS手册', price: 59, inStock: false, tags: ['语言', '类型'] }
])

function getAvailable(products: Product[]): Product[] {
  return products.filter(p => p.inStock)
}

const available = computed(() => getAvailable(products.value))
// available 类型为 ComputedRef<Product[]>
</script>

<template>
  <div>
    <div v-for="product in available" :key="product.id">
      <h3>{{ product.name }}</h3>
      <p>¥{{ product.price }}</p>
      <span v-for="tag in product.tags" :key="tag">{{ tag }}</span>
    </div>
  </div>
</template>
```

### 类型别名与联合类型

```vue
<script setup lang="ts">
// 类型别名
type Status = 'idle' | 'loading' | 'success' | 'error'
type Size = 'small' | 'medium' | 'large'

const status = ref<Status>('idle')
const size = ref<Size>('medium')

// 函数类型
type Handler = (event: MouseEvent) => void

const handleClick: Handler = (event) => {
  console.log('点击位置:', event.clientX, event.clientY)
  // 输出：点击位置: 100 200
}

// 泛型函数
function firstItem<T>(arr: T[]): T | undefined {
  return arr[0]
}

const num = firstItem([1, 2, 3])    // 类型: number | undefined
const str = firstItem(['a', 'b'])   // 类型: string | undefined
</script>
```

### 多个 script 块

```vue
<!-- 可以同时使用 lang="ts" 和普通 script -->
<script lang="ts">
// 额外的 defineComponent 配置（不常用）
export default {
  name: 'MyComponent'
}
</script>

<script setup lang="ts">
// 主要逻辑
const data = ref<string>('hello')
</script>
```

## 三、注意事项与常见陷阱

1. **`<script setup>` 中的类型会被编译时擦除**：运行时没有类型信息
2. **interface 和 type 都可用**：interface 更适合对象形状，type 更灵活
3. **泛型在 SFC 中完全支持**：可在 ref、computed、函数中使用泛型
4. **不要在模板中使用类型断言**：模板只能访问 setup 暴露的变量
5. **外部类型定义放在 .ts 文件中**：复杂接口建议提取到 `types/` 目录
