# 跨层级 Provide-Inject

## 一、概念说明
`provide` 和 `inject` 是 Vue 提供的**依赖注入**机制，用于祖先组件向任意深度的后代组件传递数据，而不需要逐层传递 props（避免"props 层层透传"问题）。

## 二、具体用法

### 2.1 基本用法
```vue
<!-- 祖先组件 -->
<script setup>
import { provide } from 'vue'

provide('appTheme', 'dark')
provide('appVersion', '1.0.0')
</script>
```

```vue
<!-- 任意后代组件 -->
<script setup>
import { inject } from 'vue'

const theme = inject('appTheme', 'light')  // 第二个参数是默认值
const version = inject('appVersion')
</script>

<template>
  <div :class="theme">当前主题: {{ theme }}，版本: {{ version }}</div>
</template>
```

### 2.2 使用 Symbol 作为 key
```ts
// keys.ts
import type { InjectionKey, Ref } from 'vue'

export const THEME_KEY: InjectionKey<Ref<string>> = Symbol('theme')
export const USER_KEY: InjectionKey<Ref<User>> = Symbol('user')
```

```vue
<script setup>
import { ref, provide } from 'vue'
import { THEME_KEY } from './keys'
provide(THEME_KEY, ref('dark'))
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
<script setup>
const showMessage = inject('showMessage')
</script>

<template>
  <button @click="showMessage('你好!')">弹出消息</button>
</template>
```

### 2.4 完整的依赖注入模式
```vue
<script setup>
import { provide, ref, readonly } from 'vue'

const state = ref({ count: 0, user: null })

// 只读数据
provide('state', readonly(state))

// 操作方法
provide('actions', {
  increment: () => state.value.count++,
  setUser: (user) => { state.value.user = user }
})
</script>
```

## 三、常见用例

| 场景 | 说明 |
|------|------|
| 主题系统 | 全局切换暗色/亮色主题 |
| 国际化 | 注入翻译函数 |
| 权限控制 | 注入权限检查方法 |
| 组件库 | 注入配置选项 |

## 四、注意事项与常见陷阱

- provide/inject 绑定**不是响应式的**（除非传递 ref/reactive）
- 尽量使用 Symbol 作为 key 避免命名冲突
- 不要过度使用 provide/inject，它使组件间依赖关系变得隐式
- 使用 `readonly()` 保护 provide 的数据，防止后代意外修改
- 传递 `ref` 时，后代组件会自动解包（不需要 `.value`）
- inject 的值可能为 undefined（未找到提供者），需要处理
