# SSR中的状态管理

## 一、概念说明

SSR 中的状态管理需要解决**状态同步**问题：服务端渲染时获取的数据必须传递给客户端，否则客户端 hydration 时数据丢失导致页面闪烁。Nuxt 通过自动序列化将服务端状态注入 HTML 的 `<script>` 标签，客户端启动时自动恢复。

## 二、具体用法

### Nuxt 自动状态序列化

```vue
<script setup>
// Nuxt 自动处理：服务端获取的数据被序列化到 window.__NUXT__
const { data: user } = await useFetch('/api/user')

// 服务端渲染时：user.value = { name: '张三', age: 25 }
// HTML 中注入：<script>window.__NUXT__ = {data:{user:{name:'张三',age:25}}}</script>
// 客户端 hydration：直接读取 window.__NUXT__，不重复请求
</script>

<template>
  <div>
    <h1>{{ user?.name }}</h1>
    <p>年龄: {{ user?.age }}</p>
  </div>
</template>
```

### 使用 useState 管理共享状态

```ts
// composables/useCart.ts
export function useCart() {
  const cart = useState('cart', () => ({
    items: [] as { id: number; name: string; qty: number }[],
    total: 0
  }))

  function addItem(item: { id: number; name: string; price: number }) {
    const existing = cart.value.items.find(i => i.id === item.id)
    if (existing) {
      existing.qty++
    } else {
      cart.value.items.push({ ...item, qty: 1 })
    }
    cart.value.total = cart.value.items.reduce(
      (sum, i) => sum + i.price * i.qty, 0
    )
  }

  return { cart, addItem }
}
```

```vue
<script setup>
const { cart, addItem } = useCart()
</script>

<template>
  <div>
    <h2>购物车 ({{ cart.items.length }} 件)</h2>
    <ul>
      <li v-for="item in cart.items" :key="item.id">
        {{ item.name }} x{{ item.qty }}
      </li>
    </ul>
    <p>总计: ¥{{ cart.total }}</p>
    <button @click="addItem({ id: 1, name: 'Vue课程', price: 99 })">
      添加商品
    </button>
  </div>
</template>
```

### Pinia SSR 状态同步

```ts
// stores/counter.ts
export const useCounterStore = defineStore('counter', () => {
  const count = ref(0)

  // onServerPrefetch 在服务端渲染时执行
  onServerPrefetch(async () => {
    // 服务端获取初始值
    count.value = 42
    // Pinia 自动将状态序列化到 HTML
  })

  function increment() {
    count.value++
  }

  return { count, increment }
})
```

```vue
<script setup>
const counter = useCounterStore()
// 服务端 count = 42
// 客户端 hydration 后 count 仍为 42（从 window.__NUXT__ 恢复）
</script>

<template>
  <div>
    <p>计数: {{ counter.count }}</p>
    <button @click="counter.increment">+1</button>
  </div>
</template>
```

## 三、注意事项与常见陷阱

1. **避免存储不可序列化对象**：函数、DOM 节点、Map、Set 不能序列化，会导致 SSR 失败
2. **每个请求的 state 隔离**：SSR 中不同用户请求有独立的状态空间
3. **序列化大小影响性能**：过大的状态对象会增加 HTML 体积，拖慢页面加载
4. **不要在服务端读取 localStorage**：这是浏览器 API，服务端不存在
5. **SSR 与 CSR 状态可能短暂不一致**：在 hydration 完成前，用户看到的是 SSR 状态
