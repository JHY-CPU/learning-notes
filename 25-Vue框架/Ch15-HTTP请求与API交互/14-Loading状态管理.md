# Loading 状态管理

## 一、概念说明

良好的加载状态反馈是用户体验的关键。常见方案包括：Loading 指示器、骨架屏（Skeleton）、占位符等。

```vue
<script setup>
import { ref } from 'vue'
import axios from 'axios'

const data = ref(null)
const loading = ref(false)

async function fetchData() {
  loading.value = true
  try {
    const res = await axios.get('/api/article')
    data.value = res.data
  } finally {
    loading.value = false
  }
}
</script>

<template>
  <button @click="fetchData">加载文章</button>

  <!-- 骨架屏 -->
  <div v-if="loading" class="skeleton">
    <div class="skeleton-line w-60"></div>
    <div class="skeleton-line w-100"></div>
    <div class="skeleton-line w-80"></div>
  </div>

  <!-- 真实内容 -->
  <article v-else-if="data">
    <h2>{{ data.title }}</h2>
    <p>{{ data.content }}</p>
  </article>
</template>

<style>
.skeleton-line {
  height: 16px;
  background: linear-gradient(90deg, #f0f0f0 25%, #e0e0e0 50%, #f0f0f0 75%);
  background-size: 200% 100%;
  animation: shimmer 1.5s infinite;
  border-radius: 4px;
  margin: 8px 0;
}
.w-60 { width: 60%; }
.w-80 { width: 80%; }
.w-100 { width: 100%; }
@keyframes shimmer {
  0% { background-position: 200% 0; }
  100% { background-position: -200% 0; }
}
</style>
```

## 二、具体用法

### 2.1 全局 Loading

```js
// stores/loading.js
import { defineStore } from 'pinia'

export const useLoadingStore = defineStore('loading', {
  state: () => ({ count: 0 }),
  getters: { isLoading: (s) => s.count > 0 },
  actions: {
    show() { this.count++ },
    hide() { this.count = Math.max(0, this.count - 1) }
  }
})
```

### 2.2 请求拦截器自动管理

```js
const loadingStore = useLoadingStore()

request.interceptors.request.use(config => {
  if (config.loading !== false) loadingStore.show()
  return config
})

request.interceptors.response.use(
  res => { loadingStore.hide(); return res },
  err => { loadingStore.hide(); return Promise.reject(err) }
)
```

### 2.3 列表 Loading 占位

```vue
<div v-if="loading" v-for="i in 5" :key="i" class="skeleton-card" />
<div v-else v-for="item in list" :key="item.id">{{ item.name }}</div>
```

## 三、注意事项与常见陷阱

- Loading 状态要覆盖所有请求路径（成功、失败、超时）
- 使用引用计数支持并发请求场景
- 骨架屏的布局应尽量接近真实内容，减少视觉跳动
