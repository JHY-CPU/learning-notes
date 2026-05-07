# Vite 构建工具详解

## 一、概念说明

Vite（法语意为"快速"）是面向现代浏览器的前端构建工具。开发时利用浏览器原生 ESM 按需编译模块，无需打包，冷启动极快。生产环境使用 Rollup 打包，输出高度优化的静态资源。

核心特性：**极速冷启动**（跳过打包）、**即时 HMR**（毫秒级热更新）、**按需编译**（只编译当前页面用到的模块）。

```vue
<script setup>
// Vite 自动处理 .vue 文件的编译
// HMR 时只更新修改的组件，不刷新页面
import { ref } from 'vue'
const msg = ref('Vite 超快！')
</script>

<template>
  <h1>{{ msg }}</h1>
</template>
```

## 二、具体用法

### 2.1 vite.config.js 配置

```js
import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import { resolve } from 'path'

export default defineConfig({
  plugins: [vue()],
  resolve: {
    alias: {
      '@': resolve(__dirname, 'src')
    }
  },
  server: {
    port: 3000,
    open: true,
    proxy: {
      '/api': {
        target: 'http://localhost:8080',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, '')
      }
    }
  },
  build: {
    outDir: 'dist',
    minify: 'terser',
    rollupOptions: {
      output: {
        manualChunks: {
          vendor: ['vue', 'vue-router', 'pinia']
        }
      }
    }
  }
})
```

### 2.2 环境变量

```bash
# .env（所有环境）
VITE_APP_TITLE=我的应用

# .env.production（生产环境）
VITE_API_BASE=https://api.example.com
```

```js
// 使用环境变量
console.log(import.meta.env.VITE_APP_TITLE)
console.log(import.meta.env.MODE) // development / production
```

### 2.3 静态资源处理

```vue
<script setup>
import logoUrl from '@/assets/logo.png'
import jsonData from '@/data/config.json'
</script>

<template>
  <img :src="logoUrl" alt="Logo" />
</template>
```

## 三、注意事项与常见陷阱

- 环境变量必须以 `VITE_` 开头才能在客户端访问
- Vite 只支持现代浏览器，不支持 IE
- CommonJS 模块需转为 ESM 或在 `optimizeDeps.include` 中配置
- HMR 不生效时检查是否用了 `defineComponent` 而非 `<script setup>`
- 生产构建使用 Rollup，开发使用 esbuild，插件可能有兼容差异
