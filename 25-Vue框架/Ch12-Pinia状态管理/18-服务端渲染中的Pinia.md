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

## 三、注意事项与常见陷阱

1. 服务端每个请求创建新的Pinia实例
2. 不要在服务端访问`localStorage`、`window`
3. 状态序列化时注意循环引用和日期对象
4. 客户端hydration时需恢复服务端状态
5. Pinia对SSR有官方支持，参考文档配置
