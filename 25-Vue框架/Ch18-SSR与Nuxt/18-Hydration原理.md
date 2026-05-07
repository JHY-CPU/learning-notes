# Hydration原理

## 一、概念说明

Hydration（客户端激活/注水）是 SSR 的关键步骤：浏览器收到服务端渲染的 HTML 并展示后，客户端 Vue 在已有 DOM 上"激活"交互能力。Vue 遍历现有 DOM 节点，将虚拟 DOM 与实际 DOM 对应，绑定事件处理器和响应式数据，而不重新创建 DOM。

## 二、具体用法

### Hydration 过程

```js
// 客户端入口
import { createSSRApp } from 'vue'
import App from './App.vue'

// createSSRApp 自动启用 hydration 模式
// 与 createApp 不同：createApp 会重新渲染，createSSRApp 会复用 DOM
const app = createSSRApp(App)
app.mount('#app')

// Hydration 步骤：
// 1. Vue 读取服务端渲染的 DOM 树
// 2. 在内存中构建虚拟 DOM
// 3. 逐节点对比虚拟 DOM 与实际 DOM
// 4. 绑定事件监听器（click、input 等）
// 5. 建立响应式数据连接
// 6. 页面变为可交互
```

### Hydration 不匹配示例

```vue
<script setup>
import { ref } from 'vue'

// 错误：服务端和客户端生成不同值
const timestamp = ref(Date.now())
// 服务端：1715000000000
// 客户端：1715000001234
// → Hydration 警告：text content mismatch
</script>

<template>
  <div>
    <!-- 服务端渲染：1715000000000 -->
    <!-- 客户端期望：1715000001234 -->
    <!-- 控制台警告：[Vue warn]: Hydration text content mismatch -->
    <p>生成时间: {{ timestamp }}</p>
  </div>
</template>
```

### 正确处理客户端专用逻辑

```vue
<script setup>
import { ref, onMounted } from 'vue'

// 方案1：使用 onMounted，在 hydration 后执行
const clientTime = ref('')
onMounted(() => {
  clientTime.value = new Date().toLocaleString()
  // 仅在客户端执行，不影响 SSR 输出
})

// 方案2：使用 ClientOnly 组件
</script>

<template>
  <div>
    <!-- 服务端不渲染，客户端才显示 -->
    <ClientOnly>
      <p>当前时间: {{ clientTime }}</p>
      <template #fallback>
        <p>加载中...</p>
      </template>
    </ClientOnly>

    <!-- 静态内容正常 SSR -->
    <h1>欢迎页面</h1>
  </div>
</template>
```

### Hydration 性能优化

```vue
<script setup>
// 使用 lazy hydration 延迟非关键组件的激活
const LazyHeavy = defineAsyncComponent(() =>
  import('./HeavyComponent.vue')
)
</script>

<template>
  <div>
    <!-- 首屏内容立即 hydration -->
    <HeroSection />
    <Navigation />

    <!-- 非首屏内容延迟 hydration -->
    <LazyHeavy />
    <!-- 用户滚动到视口时才激活 -->
  </div>
</template>
```

## 三、注意事项与常见陷阱

1. **Hydration 不匹配会导致警告**：控制台出现黄色警告，功能仍可用但可能有闪烁
2. **不要在模板中使用随机数**：`Math.random()` 在服务端和客户端结果不同
3. **HTML 结构必须完全一致**：包括空白文本节点的差异都会触发警告
4. **条件渲染要谨慎**：`v-if` 在服务端和客户端的判断条件必须相同
5. **Hydration 完成前页面不可交互**：用户点击按钮无响应，需用 loading 状态提示
