# Tree Shaking

## 一、概念说明

Tree Shaking 是通过静态分析移除 JavaScript 中未使用代码的优化技术。Vite（基于 Rollup/esbuild）会自动进行 Tree Shaking，但需要代码使用 ES Module 语法。

```js
// utils.js - 良好的导出方式
export function formatDate(date) { /* ... */ }
export function formatCurrency(num) { /* ... */ }
export function debounce(fn, delay) { /* ... */ }

// 使用方只导入需要的
import { formatDate } from './utils'
// debounce 和 formatCurrency 不会被打包
```

## 二、具体用法

### 2.1 确保 Tree Shaking 生效

```js
// ✅ 具名导入（支持 Tree Shaking）
import { ref, computed } from 'vue'
import { formatDate } from '@/utils'

// ❌ 全量导入（阻止 Tree Shaking）
import * as utils from '@/utils'
import _ from 'lodash'
```

### 2.2 package.json 配置

```json
{
  "sideEffects": false
}
```

```json
// 有副作用的文件要排除
{
  "sideEffects": [
    "*.css",
    "*.vue",
    "./src/polyfills.js"
  ]
}
```

### 2.3 第三方库的 Tree Shaking

```js
// lodash - 使用 lodash-es（ES Module 版本）
import { debounce } from 'lodash-es'

// Element Plus - 按需导入
import { ElButton, ElInput } from 'element-plus'

// Day.js - 按需导入插件
import dayjs from 'dayjs'
import relativeTime from 'dayjs/plugin/relativeTime'
dayjs.extend(relativeTime)
```

### 2.4 分析打包结果

```bash
# vite-plugin-visualizer
npm install -D vite-plugin-visualizer
```

```js
// vite.config.js
import { visualizer } from 'rollup-plugin-visualizer'

export default defineConfig({
  plugins: [
    visualizer({ open: true, gzipSize: true })
  ]
})
```

## 四、常见库的 Tree Shaking 最佳实践

```js
// lodash → lodash-es（ESM 版本）
// ❌
import _ from 'lodash'
_.debounce(fn, 300)

// ✅
import { debounce } from 'lodash-es'

// dayjs → 具名导入插件
// ❌
import dayjs from 'dayjs'
import 'dayjs/locale/zh-cn'  // 副作用导入

// ✅
import dayjs from 'dayjs'
import zhCn from 'dayjs/locale/zh-cn'  // 具名导入
dayjs.locale(zhCn)

// Element Plus → 自动导入
// 使用 unplugin-vue-components + unplugin-auto-import
// 无需手动 import，按需自动导入

// Ant Design Vue → 按需导入
import { Button, Input } from 'ant-design-vue'
```

## 五、检查 Tree Shaking 效果

```bash
# 安装分析工具
npm install -D rollup-plugin-visualizer

# 构建后查看
npx vite build
# 打开生成的 stats.html
```

```js
// vite.config.js
import { visualizer } from 'rollup-plugin-visualizer'

export default defineConfig({
  plugins: [
    visualizer({
      open: true,
      gzipSize: true,
      brotliSize: true
    })
  ]
})
```

## 六、Vue 生态的 Tree Shaking

```js
// Vue 3 天然支持 Tree Shaking
import { ref, computed } from 'vue'
// 未使用的 API（如 watch、reactive）不会打包

// Vue Router
import { createRouter, createWebHistory } from 'vue-router'
// 只打包使用的功能

// Pinia
import { defineStore } from 'pinia'
// 只打包定义的 store

// VueUse
import { useDebounceFn } from '@vueuse/core'
// 只打包 useDebounceFn 及其依赖
```

## 三、注意事项与常见陷阱

- 使用 ES Module 语法（`import/export`），避免 CommonJS（`require`）
- 避免 `import *` 和默认导入整个库
- 确保 `package.json` 中 `sideEffects` 配置正确
- 某些库（如 lodash v4）需要使用 `lodash-es` 或 `xxx/xxx` 路径导入
- 自动导入插件（unplugin-auto-import）可以减少代码量但不影响 Tree Shaking
- CSS 文件要标记为 sideEffects，否则可能被误删
