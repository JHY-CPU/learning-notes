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


<!-- Converted from: 76_性能-Tree Shaking.html -->
