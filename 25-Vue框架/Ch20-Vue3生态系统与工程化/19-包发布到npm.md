# 包发布到npm

## 一、概念说明

将 Vue 3 组件库或工具包发布到 npm 需要：打包为标准格式（ESM/CJS）、生成类型声明、配置 package.json 的 exports 字段、使用 changesets 管理版本。Vite 的 library mode 简化了打包流程。

## 二、具体用法

### Vite Library Mode 配置

```ts
// vite.config.ts
import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import dts from 'vite-plugin-dts'

export default defineConfig({
  plugins: [
    vue(),
    dts({ include: ['src/**/*.ts', 'src/**/*.vue'] })
  ],
  build: {
    lib: {
      entry: './src/index.ts',
      name: 'MyLib',
      formats: ['es', 'cjs', 'umd'],
      fileName: (format) => `my-lib.${format}.js`
    },
    rollupOptions: {
      external: ['vue'],
      output: {
        globals: { vue: 'Vue' }
      }
    }
  }
})
```

### 组件库入口

```ts
// src/index.ts
import type { App } from 'vue'
import MyButton from './components/MyButton.vue'
import MyCard from './components/MyCard.vue'

// Vue 插件方式安装
export function install(app: App) {
  app.component('MyButton', MyButton)
  app.component('MyCard', MyCard)
}

// 按需导出
export { MyButton, MyCard }

// 类型导出
export type { MyButtonProps } from './components/MyButton.vue'
```

### package.json 配置

```json
{
  "name": "@my-org/vue-components",
  "version": "1.0.0",
  "type": "module",
  "main": "./dist/my-lib.cjs.js",
  "module": "./dist/my-lib.es.js",
  "types": "./dist/index.d.ts",
  "exports": {
    ".": {
      "import": "./dist/my-lib.es.js",
      "require": "./dist/my-lib.cjs.js",
      "types": "./dist/index.d.ts"
    },
    "./style.css": "./dist/style.css"
  },
  "files": ["dist"],
  "sideEffects": ["**/*.css"],
  "peerDependencies": {
    "vue": "^3.3.0"
  },
  "scripts": {
    "build": "vite build",
    "release": "npm run build && npm publish"
  }
}
```

### 版本管理

```bash
# 安装 changesets
npm install -D @changesets/cli
npx changeset init

# 发布流程
npx changeset          # 记录变更
npx changeset version  # 更新版本号
npm run build
npm publish --access public
```

```markdown
<!-- .changeset/my-change.md -->
---
"@my-org/vue-components": minor
---

新增 MyCard 组件
```

### 用户使用

```ts
// 用户安装
// npm install @my-org/vue-components

// 全量引入
import { install } from '@my-org/vue-components'
import '@my-org/vue-components/style.css'
app.use(install)

// 按需引入
import { MyButton } from '@my-org/vue-components'
```

## 三、注意事项与常见陷阱

1. **Vue 必须设为 peerDependencies**：不能打包进组件库
2. **CSS 需要单独导出**：样式文件在 exports 中声明
3. **types 字段必须正确**：否则用户无法获得类型提示
4. **files 字段控制发布内容**：只发布 dist 目录，排除源码
5. **scoped 包需要 --access public**：`@org/` 前缀的包默认受限
