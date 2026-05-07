# 跨层级 Provide-Inject

## 一、概念说明
`provide` 和 `inject` 是 Vue 提供的**依赖注入**机制，用于祖先组件向任意深度的后代组件传递数据，而不需要逐层传递 props（避免"props 层层透传"问题）。

## 二、具体用法

### 2.1 基本用法
```vue
<!-- 祖先组件 -->
<script setup>
import { provide } from 'vue'
import Child from './Child.vue'

provide('appTheme', 'dark')
provide('appVersion', '1.0.0')
</script>
```

```vue
<!-- 任意后代组件（不需要是直接子组件） -->
<script setup>
import { inject } from 'vue'

const theme = inject('appTheme', 'light')  // 第二个参数是默认值
const version = inject('appVersion')
</script>

<template>
  <div :class="theme">当前主题: {{ theme }}，版本: {{ version }}</div>
</template>
```

### 2.2 传递非字符串 key
```vue
<script setup>
import { provide, InjectionKey, Ref } from 'vue'

// TypeScript 中使用 InjectionKey 做类型约束
const themeKey: InjectionKey<Ref<string>> = Symbol('theme')
provide(themeKey, ref('dark'))

// 后代组件
const theme = inject(themeKey)
</script>
```

### 2.3 传递函数实现后代调用祖先方法
```vue
<script setup>
import { provide } from 'vue'

function showMessage(msg) {
  alert(msg)
}
provide('showMessage', showMessage)
</script>
```

```vue
<!-- 后代组件 -->
<script setup>
const showMessage = inject('showMessage')
</script>

<template>
  <button @click="showMessage('你好!')">弹出消息</button>
</template>
```

## 三、注意事项与常见陷阱
- provide/inject 绑定**不是响应式的**（除非传递 ref/reactive）
- 尽量使用 Symbol 作为 key 避免命名冲突
- 不要过度使用 provide/inject，它使组件间依赖关系变得隐式
- 与 React 的 Context 类似，但 Vue 的实现更轻量
