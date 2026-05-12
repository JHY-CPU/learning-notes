# 性能-Tree Shaking


## 性能-Tree Shaking


ES 模块静态分析、sideEffects 标记、消除 dead code、打包优化。


## Tree Shaking API


```
// ========== Tree Shaking 原理 ==========
// 基于 ES Module 静态分析 (import/export)
// 在打包时移除未使用的导出

// ========== 前提条件 ==========
// 1. ES Module (import/export)
// 2. sideEffects: false (package.json)
// 3. 生产模式 (mode: 'production')

// ========== 支持 Tree Shaking 的导入方式 ==========
import { debounce } from 'lodash-es';        // ✅ 按需
import * as utils from './utils';            // ❌ 全部
import _ from 'lodash';                      // ❌ 全部 (CJS)
```


## 演示：Tree Shaking

点击按钮查看


## 什么是 Tree Shaking

Tree Shaking 是在打包过程中自动移除未使用代码（dead code）的优化技术。依赖ES Module的静态分析能力，在构建时确定哪些导出没有被使用。

## 工作原理

1. 分析所有 `import` 语句，标记被使用的导出
2. 未被导入的导出标记为"未使用"
3. 在生产构建时移除这些未使用的导出
4. 配合压缩工具（Terser）进一步清除dead code

## 前提条件

1. **使用ES Module**：`import` / `export`（CommonJS的 `require` 不支持静态分析）
2. **package.json 标记**：`"sideEffects": false` 告诉打包工具该包无副作用，可安全删除未使用的导出
3. **生产模式**：Webpack需 `mode: 'production'`，Vite默认生产模式开启

## 最佳实践

```js
// ✅ 按需导入
import { debounce } from 'lodash-es';

// ❌ 全量导入（无法tree shake）
import _ from 'lodash';

// ✅ 使用babel-plugin-import自动转换
```

## 常见问题

- CommonJS模块（如 `lodash`）不支持tree shake，使用 `lodash-es` 替代
- 有副作用的代码（如CSS导入、polyfill）需在 `sideEffects` 中声明
- Webpack的 `usedExports` 和 `sideEffects` 两个选项配合使用
- Rollup天然支持tree shaking，比Webpack更彻底

<!-- Converted from: 76_性能-Tree Shaking.html -->
