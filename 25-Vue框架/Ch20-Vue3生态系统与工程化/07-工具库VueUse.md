# 工具库VueUse

## 一、概念说明

VueUse 是 Vue 组合式 API 工具函数集合，提供 200+ 实用的 Composable。覆盖浏览器 API、状态、动画、网络、传感器等场景。它是 Vue 生态中最受欢迎的工具库，几乎所有 Vue 3 项目都会用到。

## 二、具体用法

### 安装与基本使用

```bash
npm install @vueuse/core
```

### 常用函数

```vue
<script setup lang="ts">
import {
  useMouse,
  useLocalStorage,
  useDebounceFn,
  useEventListener,
  useIntersectionObserver,
  useClipboard,
  useMediaQuery,
  useDark,
  useToggle
} from '@vueuse/core'

// 鼠标位置
const { x, y } = useMouse()
// x, y 是 Ref<number>，实时跟踪鼠标位置

// 本地存储（响应式）
const theme = useLocalStorage('theme', 'light')
theme.value = 'dark'
// 自动同步到 localStorage

// 防抖
const search = ref('')
const debouncedSearch = useDebounceFn(() => {
  console.log('搜索:', search.value)
}, 300)

// 剪贴板
const { copy, copied, text } = useClipboard()
function copyText() {
  copy('要复制的文本')
  // copied.value 变为 true
}

// 暗色模式
const isDark = useDark()
const toggleDark = useToggle(isDark)

// 响应式媒体查询
const isMobile = useMediaQuery('(max-width: 768px)')
// isMobile.value → true/false

// 元素进入视口
const target = ref<HTMLDivElement | null>(null)
const { stop } = useIntersectionObserver(target, ([entry]) => {
  if (entry?.isIntersecting) {
    console.log('元素进入视口')
    stop() // 只触发一次
  }
}, { threshold: 0.5 })
</script>

<template>
  <div>
    <p>鼠标位置: {{ x }}, {{ y }}</p>
    <p>当前主题: {{ theme }}</p>
    <p>是否移动端: {{ isMobile }}</p>
    <input v-model="search" @input="debouncedSearch" placeholder="搜索" />
    <button @click="toggleDark()">切换暗色</button>
    <button @click="copyText">
      {{ copied ? '已复制' : '复制文本' }}
    </button>
    <div ref="target" style="height: 200px; margin-top: 500px">
      滚动到此处触发
    </div>
  </div>
</template>
```

### 网络与传感器

```vue
<script setup lang="ts">
import { useFetch, useOnline, useGeolocation, useBattery } from '@vueuse/core'

// 网络状态
const isOnline = useOnline()
// isOnline.value → true/false

// 电池状态
const { charging, level } = useBattery()
// level.value → 0.85 (85%)

// 地理位置
const { coords, locatedAt, error } = useGeolocation()
// coords.value.latitude, coords.value.longitude

// 封装的 fetch
const { data, isFetching, error: fetchError } = useFetch('/api/data', {
  refetch: true
}).json<{ name: string }>()
</script>

<template>
  <div>
    <p>网络: {{ isOnline ? '在线' : '离线' }}</p>
    <p>电量: {{ Math.round((level ?? 0) * 100) }}% {{ charging ? '(充电中)' : '' }}</p>
    <div v-if="data">
      <h2>{{ data.name }}</h2>
    </div>
  </div>
</template>
```

## 三、注意事项与常见陷阱

1. **所有函数都返回 Ref**：在模板中自动解包，JS 中用 `.value`
2. **useFetch 不替代 axios**：复杂请求场景仍推荐 axios 或 ofetch
3. **SSR 兼容性**：VueUse 函数大多 SSR 友好，自动检测环境
4. **按需导入**：`import { useMouse } from '@vueuse/core'` 不会引入全部函数
5. **useIntersectionObserver 需要 polyfill**：旧浏览器不支持 IntersectionObserver
