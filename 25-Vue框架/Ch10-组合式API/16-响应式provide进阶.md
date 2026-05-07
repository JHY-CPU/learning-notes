# 响应式provide进阶

## 一、概念说明

默认情况下`provide/inject`不是响应式的。要让注入的值响应变化，需要提供`ref`或`reactive`对象。使用`readonly`包装可防止后代意外修改。

```vue
<!-- 祖先组件 -->
<script setup>
import { provide, ref, readonly } from 'vue'

const count = ref(0)
const increment = () => count.value++

// 提供只读的ref + 修改方法
provide('count', readonly(count))
provide('increment', increment)
</script>
```

## 二、具体用法

### 响应式状态 + 只读保护

```vue
<!-- StoreProvider.vue -->
<script setup>
import { provide, reactive, readonly, toRefs } from 'vue'

const state = reactive({
  user: null,
  theme: 'light',
  notifications: []
})

const setUser = (user) => { state.user = user }
const toggleTheme = () => {
  state.theme = state.theme === 'light' ? 'dark' : 'light'
}

provide('store', readonly(state))
provide('setUser', setUser)
provide('toggleTheme', toggleTheme)
</script>

<!-- 任意后代 -->
<script setup>
import { inject } from 'vue'

const store = inject('store')
const toggleTheme = inject('toggleTheme')

// store是只读的，直接修改会警告
// toggleTheme可以安全地修改状态
</script>
```

### 模块化provide

```js
// providers/themeProvider.js
import { ref, readonly, provide, inject } from 'vue'

const THEME_KEY = Symbol('theme')

export function provideTheme() {
  const theme = ref('light')
  const toggle = () => {
    theme.value = theme.value === 'light' ? 'dark' : 'light'
  }
  provide(THEME_KEY, { theme: readonly(theme), toggle })
}

export function useTheme() {
  const context = inject(THEME_KEY)
  if (!context) throw new Error('useTheme必须在ThemeProvider内使用')
  return context
}
```

## 三、注意事项与常见陷阱

1. 使用`readonly`包装后，修改会触发控制台警告但不会生效
2. `readonly`是浅层的，嵌套对象仍可被修改，需用`deepReadonly`
3. 响应式provide适用于应用级状态，简单场景可用，复杂场景建议Pinia
4. 每次provide都会覆盖同key的祖先值
5. 提供工厂函数（修改方法）比提供可写状态更安全
