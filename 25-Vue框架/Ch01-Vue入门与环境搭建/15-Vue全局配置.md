# Vue 全局配置

## 一、概念说明

Vue 3 通过 `app.config` 对象提供全局配置能力。可以设置全局错误处理器、全局属性、性能追踪等。全局配置影响整个应用实例。

```vue
<script setup>
// main.js 中的全局配置
import { createApp } from 'vue'
import App from './App.vue'

const app = createApp(App)

// 全局错误处理
app.config.errorHandler = (err, instance, info) => {
  console.error('全局错误:', err)
  console.log('错误信息:', info)
}

// 全局属性（Vue 3 推荐用 provide/inject 替代）
app.config.globalProperties.$http = fetch
app.config.globalProperties.$appName = '我的应用'

app.mount('#app')
</script>
```

## 二、具体用法

### 2.1 errorHandler 错误处理

```js
app.config.errorHandler = (err, vm, info) => {
  // err: 错误对象
  // vm: 发生错误的组件实例
  // info: Vue 特定的错误信息（如生命周期钩子名）

  // 上报到错误监控服务
  reportError({ error: err, component: vm?.$options?.name, info })
}
```

### 2.2 warnHandler 警告处理

```js
// 仅开发环境生效
app.config.warnHandler = (msg, vm, trace) => {
  console.warn('Vue 警告:', msg)
  console.warn('组件追踪:', trace)
}
```

### 2.3 performance 性能追踪

```js
app.config.performance = true // 开发环境自动启用
// 现在可以在 DevTools 中看到组件渲染耗时
```

### 2.4 全局属性 vs provide/inject

```js
// 方式1: globalProperties（不推荐用于新项目）
app.config.globalProperties.$utils = { formatDate, formatCurrency }

// 方式2: provide（推荐）
app.provide('utils', { formatDate, formatCurrency })
```

## 三、注意事项与常见陷阱

- `globalProperties` 在 TypeScript 中没有类型推断，建议用 `provide/inject`
- `errorHandler` 无法捕获异步错误（需配合 `window.onerror`）
- `performance` 仅开发环境有效，生产环境自动关闭
- 全局属性会与组件内同名属性冲突（组件优先）
- 配置应在 `app.mount()` 之前完成
