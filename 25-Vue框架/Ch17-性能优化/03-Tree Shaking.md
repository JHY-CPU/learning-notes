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

## 三、注意事项与常见陷阱

- 使用 ES Module 语法（`import/export`），避免 CommonJS（`require`）
- 避免 `import *` 和默认导入整个库
- 确保 `package.json` 中 `sideEffects` 配置正确
- 某些库（如 lodash v4）需要使用 `lodash-es` 或 `xxx/xxx` 路径导入
