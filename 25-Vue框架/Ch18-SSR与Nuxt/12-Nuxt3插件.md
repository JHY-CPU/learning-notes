# Nuxt3插件

## 一、概念说明

Nuxt 3 插件存放在 `plugins/` 目录，在应用初始化时自动执行。插件用于注册全局功能：添加 Vue 插件、配置全局属性、初始化第三方库。插件可指定在客户端或服务端运行，文件名中包含 `.client` 或 `.server` 后缀来控制执行环境。

## 二、具体用法

### 基本插件

```ts
// plugins/my-plugin.ts
export default defineNuxtPlugin((nuxtApp) => {
  // nuxtApp 是 Nuxt 应用实例
  // 提供全局工具函数
  return {
    provide: {
      // 注入 $hello 函数，全局可用
      hello: (name: string) => `你好，${name}！`
    }
  }
})
```

```vue
<!-- 任何组件中使用 -->
<script setup>
const { $hello } = useNuxtApp()
console.log($hello('Vue'))
// 控制台输出：你好，Vue！
</script>
```

### 客户端插件

```ts
// plugins/analytics.client.ts
// 仅在客户端执行，服务端跳过
export default defineNuxtPlugin(() => {
  // 安全使用浏览器 API
  window.dataLayer = window.dataLayer || []

  // 页面浏览追踪
  const router = useRouter()
  router.afterEach((to) => {
    window.dataLayer.push({
      event: 'pageview',
      page: to.fullPath
    })
    console.log('追踪页面:', to.fullPath)
  })
})
```

### 服务端插件

```ts
// plugins/api.server.ts
// 仅在服务端执行
export default defineNuxtPlugin(() => {
  // 初始化数据库连接等服务端资源
  const db = {
    query: (sql: string) => {
      console.log('执行 SQL:', sql)
      return []
    }
  }

  return {
    provide: { db }
  }
})
```

### 添加 Vue 插件

```ts
// plugins/toast.ts
import Toast from 'vue-toastification'
import 'vue-toastification/dist/index.css'

export default defineNuxtPlugin((nuxtApp) => {
  nuxtApp.vueApp.use(Toast, {
    position: 'top-right',
    timeout: 3000
  })
})
```

## 三、注意事项与常见陷阱

1. **插件按文件名排序执行**：文件名影响执行顺序，用数字前缀控制：`01-plugin.ts`、`02-plugin.ts`
2. **客户端插件不会在 SSR 中执行**：不要在客户端插件中进行影响 HTML 输出的操作
3. **`provide` 的值自动添加 `$` 前缀**：`provide: { foo: 'bar' }` 使用时为 `$foo`
4. **插件中不能使用自动导入的组件**：需要手动 import Vue 相关 API
5. **插件中不要执行耗时操作**：插件会阻塞应用初始化，影响首屏加载时间
