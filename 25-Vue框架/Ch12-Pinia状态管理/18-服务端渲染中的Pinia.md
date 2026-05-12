# 服务端渲染中的Pinia

## 一、概念说明

在SSR（服务端渲染）中，Pinia需要在服务端创建实例，序列化状态后在客户端恢复。确保服务端和客户端状态一致。

```js
// server.js
import { createPinia } from 'pinia'

// 服务端每次请求创建新的Pinia实例
const pinia = createPinia()
app.use(pinia)

// 序列化状态
const initialState = JSON.stringify(pinia.state.value)
// 注入到HTML中
html.replace('<!--pinia-state-->', initialState)
```

```js
// client.js
import { createPinia } from 'pinia'

const pinia = createPinia()

// 从服务端注入的状态恢复
if (window.__INITIAL_STATE__) {
  pinia.state.value = JSON.parse(window.__INITIAL_STATE__)
}

app.use(pinia)
```

## 二、具体用法

### 避免状态污染

```js
// 每个请求创建新的pinia实例
app.get('*', (req, res) => {
  const pinia = createPinia()  // 不要复用
  const app = createSSRApp(App)
  app.use(pinia)
  // ...
})
```

### Store中的SSR安全检查

```js
export const useAuth = defineStore('auth', () => {
  const token = ref('')

  // SSR安全读取
  const initToken = () => {
    if (import.meta.env.SSR) return
    token.value = localStorage.getItem('token') || ''
  }

  return { token, initToken }
})
```

## 四、Nuxt 3 中的 Pinia

```ts
// nuxt.config.ts
export default defineNuxtConfig({
  modules: ['@pinia/nuxt']
})

// stores/counter.ts
export const useCounter = defineStore('counter', () => {
  const count = ref(0)
  const increment = () => count.value++
  return { count, increment }
})

// 页面中自动可用，无需手动创建 pinia 实例
```

## 五、状态序列化注意事项

```js
// ❌ 不可序列化的数据
const store = defineStore('bad', () => {
  const func = () => {}        // 函数
  const el = ref(null)         // DOM元素
  const date = ref(new Date()) // Date对象序列化后变字符串
  const regex = ref(/test/)    // 正则表达式
  return { func, el, date, regex }
})

// ✅ 只存储可序列化的数据
const store = defineStore('good', () => {
  const timestamp = ref(Date.now())  // 用数字代替 Date
  const dateString = ref(new Date().toISOString())
  return { timestamp, dateString }
})
```

## 六、条件性访问浏览器API

```js
export const useTheme = defineStore('theme', () => {
  const theme = ref('light')

  const init = () => {
    if (import.meta.env.SSR) return
    theme.value = localStorage.getItem('theme') || 'light'
    document.documentElement.setAttribute('data-theme', theme.value)
  }

  const setTheme = (newTheme) => {
    theme.value = newTheme
    if (!import.meta.env.SSR) {
      localStorage.setItem('theme', newTheme)
      document.documentElement.setAttribute('data-theme', newTheme)
    }
  }

  return { theme, init, setTheme }
})
```

## 三、注意事项与常见陷阱

1. 服务端每个请求创建新的Pinia实例
2. 不要在服务端访问`localStorage`、`window`
3. 状态序列化时注意循环引用和日期对象
4. 客户端hydration时需恢复服务端状态
5. Pinia对SSR有官方支持，参考文档配置
6. 使用 `import.meta.env.SSR` 判断当前环境
7. Nuxt 3 中使用 `@pinia/nuxt` 模块自动处理 SSR
