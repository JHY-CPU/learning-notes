# Nuxt3数据获取useFetch

## 一、概念说明

Nuxt 3 提供 `useFetch`、`useAsyncData` 和 `useLazyFetch` 三个组合式函数用于数据获取。它们在服务端和客户端都能工作：SSR 时在服务端获取数据并序列化到 HTML，客户端 hydration 时直接使用缓存数据，避免重复请求。

## 二、具体用法

### useFetch 基本用法

```vue
<script setup>
// useFetch 自动处理 SSR/CSR 差异
// 服务端：渲染前获取数据 → 客户端：直接使用缓存
const { data: posts, pending, error, refresh } = await useFetch('/api/posts', {
  // 请求选项
  headers: { 'Authorization': 'Bearer token' }
})

// posts 的值：
// pending 为 true 时：null
// 请求完成后：[{ id: 1, title: '文章标题', ... }]
// 请求失败时：null，error 包含错误信息
</script>

<template>
  <div>
    <div v-if="pending">加载中...</div>
    <div v-else-if="error">错误: {{ error.message }}</div>
    <div v-else>
      <article v-for="post in posts" :key="post.id">
        <h2>{{ post.title }}</h2>
        <p>{{ post.body }}</p>
      </article>
      <button @click="refresh()">刷新数据</button>
    </div>
  </div>
</template>
```

### useAsyncData 用法

```vue
<script setup>
// useAsyncData 适合自定义获取逻辑
const { data: user } = await useAsyncData('user-profile', async () => {
  // key 为 'user-profile'，用于缓存和去重
  const res = await $fetch('/api/user/1')
  return {
    ...res,
    fullName: `${res.firstName} ${res.lastName}`
  }
})

// 输出：user.value = { id: 1, firstName: '张', lastName: '三', fullName: '张 三' }
</script>

<template>
  <div v-if="user">
    <h1>{{ user.fullName }}</h1>
    <p>ID: {{ user.id }}</p>
  </div>
</template>
```

### useLazyFetch 懒加载

```vue
<script setup>
// useLazyFetch 不阻塞导航，数据在组件挂载后获取
const { data: comments, pending } = await useLazyFetch('/api/comments')

// 页面立即渲染，comments 初始为 null
// 数据获取完成后自动更新视图
</script>

<template>
  <div>
    <h2>评论区</h2>
    <!-- 数据获取前显示骨架屏 -->
    <div v-if="pending" class="skeleton">加载评论中...</div>
    <ul v-else>
      <li v-for="c in comments" :key="c.id">{{ c.text }}</li>
    </ul>
  </div>
</template>
```

### $fetch 直接调用

```vue
<script setup>
// $fetch 是 Nuxt 封装的请求工具，可在服务端和客户端使用
async function submitForm(formData) {
  const result = await $fetch('/api/posts', {
    method: 'POST',
    body: formData
  })
  console.log('创建成功:', result)
  // 输出：创建成功: { id: 10, title: '新文章', ... }
}
</script>
```

## 三、注意事项与常见陷阱

1. **useFetch 必须在 setup 中使用**：不能在普通函数或条件语句中调用
2. **key 唯一性**：`useAsyncData` 的第一个参数是缓存 key，重复 key 会导致数据覆盖
3. **数据是 ref**：`data` 返回的是 Ref 对象，模板中自动解包，JS 中需用 `.value`
4. **useLazyFetch 不阻塞 SSR**：服务端也会获取数据，只是不等待完成就开始渲染
5. **避免在 useFetch 中使用响应式变量的 .value**：传入 getter 函数确保响应式追踪
