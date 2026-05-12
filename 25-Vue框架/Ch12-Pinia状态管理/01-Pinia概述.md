# Pinia概述

## 一、概念说明

Pinia是Vue 3的官方状态管理库，是Vuex的继任者。它提供更简洁的API、更好的TypeScript支持，且没有mutations的概念。

```js
// main.js
import { createApp } from 'vue'
import { createPinia } from 'pinia'
import App from './App.vue'

const app = createApp(App)
app.use(createPinia())
app.mount('#app')
```

```js
// stores/counter.js
import { defineStore } from 'pinia'

export const useCounterStore = defineStore('counter', {
  state: () => ({ count: 0 }),
  getters: { doubled: (state) => state.count * 2 },
  actions: {
    increment() { this.count++ }
  }
})
```

## 二、具体用法

### Pinia vs Vuex对比

| 特性 | Pinia | Vuex 4 |
|------|-------|--------|
| TypeScript | 原生支持 | 需额外配置 |
| Mutations | 无 | 有 |
| 模块 | 自动 | 需命名空间 |
| DevTools | 支持 | 支持 |
| 包大小 | ~1KB | ~3KB |
| Vue版本 | Vue 2/3 | Vue 2/3 |

### 三个核心概念

- **State**：状态数据（类似data）
- **Getters**：计算属性（类似computed）
- **Actions**：方法（同步/异步均可，替代mutations+actions）

## 三、注意事项与常见陷阱

1. Pinia需要在app上`use(pinia)`后才能使用store
2. 每个store用`defineStore`定义，第一个参数是唯一ID
3. Pinia没有`mutations`，所有修改都在`actions`中进行
4. Store是响应式的，直接修改state即可触发更新
5. Pinia支持Vue DevTools，可调试状态变化

## 四、Pinia 的核心优势

| 优势 | 说明 |
|------|------|
| 完整的 TypeScript 支持 | 自动类型推断，无需手动声明 |
| 极轻量 | ~1KB（gzip）|
| 扁平的模块系统 | 无需嵌套命名空间 |
| 无 mutations | 简化数据修改流程 |
| 支持 Composition API | Setup 函数风格 |
| 插件系统 | 可扩展 store 功能 |
| 热更新 | 开发时修改 store 不丢失状态 |

## 五、项目结构

```
src/
  stores/
    index.js          # pinia 初始化
    user.js           # 用户 store
    cart.js           # 购物车 store
    app.js            # 应用配置 store
    modules/          # 复杂 store 拆分
      auth.js
      settings.js
```
