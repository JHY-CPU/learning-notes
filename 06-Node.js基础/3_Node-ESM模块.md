# Node-ESM模块


## Node ESM 模块


.mjs 扩展名、package.json type、import/export、与 CJS 交互。


## Node.js 中的 ES 模块


```
// ========== 启用 ESM ==========
// 方式 1: .mjs 扩展名
// file.mjs → 始终是 ES module

// 方式 2: package.json 设置
// { "type": "module" }
// .js 文件被视为 ES module

// ========== package.json 配置 ==========
// {
//   "type": "module",          // .js → ESM
//   "exports": {
//     ".": "./index.js",
//     "./utils": "./utils.js"
//   },
//   "main": "./index.js"       // CJS 兼容
// }

// ========== ESM 导入 ==========
import fs from 'fs';                 // 默认导入
import { readFile } from 'fs';       // 命名导入
import path from 'node:path';        // node: 协议
import { fileURLToPath } from 'url'; // URL 转换

// ========== __dirname 替代 ==========
// ESM 中没有 __dirname, __filename
import { fileURLToPath } from 'url';
import { dirname } from 'path';
const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// ========== CJS 与 ESM 交互 ==========
// ESM 可以 import CJS 模块
import lodash from 'lodash'; // ✅ CJS → ESM

// CJS 不能 require ESM 模块 (需动态 import)
// const esm = await import('./esm-module.mjs');

// ========== 顶级 await (ESM) ==========
// ESM 支持顶级 await
const data = await fetch('https://api.example.com/data');
export default data;
```


> **Note:** 💡 ESM 是静态分析，支持 tree-shaking。
>
>
> ⚠️ CJS 和 ESM 行为不同，需注意兼容。
>
>
> 💡 Node 默认是 CJS (.js → CJS) 除非设置 type: module。


## 演示：Node ESM

点击按钮查看


## ESM 动态导入与条件导入

```javascript
// ========== 动态 import ==========
// 可以在任意位置使用 (类似 require)
async function loadConfig() {
    const config = await import('./config.js');
    return config.default;
}

// ========== 条件导入 ==========
async function loadModule(useTypeScript) {
    if (useTypeScript) {
        const ts = await import('typescript');
        return ts;
    }
    return null;
}

// ========== 按需加载 (代码分割) ==========
async function handleRequest(req) {
    if (req.url.startsWith('/api')) {
        const { apiRouter } = await import('./routes/api.js');
        return apiRouter(req);
    }
    const { pageRouter } = await import('./routes/pages.js');
    return pageRouter(req);
}
```

## ESM 导出高级用法

```javascript
// ========== 重导出 (re-export) ==========
// 从其他模块重新导出
export { foo, bar } from './utils.js';
export { default as Utils } from './utils.js';
export * from './constants.js';

// ========== 命名导出 vs 默认导出 ==========
// utils.js
export const PI = 3.14159;         // 命名导出
export function sum(a, b) { return a + b; }
export default class Utils {}      // 默认导出 (每个模块最多一个)

// 导入时
import Utils, { PI, sum } from './utils.js';

// ========== 导入全部 ==========
import * as All from './utils.js';
console.log(All.PI, All.sum(1, 2));
```

## ESM 与 CJS 互操作注意事项

```javascript
// ========== ESM 导入 CJS ==========
import fs from 'fs';                    // ✅ 默认导入
import { readFile } from 'fs';          // ✅ 命名导入
const lodash = (await import('lodash')).default; // ✅ 动态导入

// ========== CJS 导入 ESM (需要动态 import) ==========
// common.cjs
async function loadESM() {
    const { something } = await import('./esm-module.mjs');
    return something;
}

// ========== package.json exports 映射 ==========
// {
//   "exports": {
//     ".": {
//       "import": "./dist/index.mjs",    // ESM 入口
//       "require": "./dist/index.cjs",   // CJS 入口
//       "default": "./dist/index.js"
//     }
//   }
// }

// ========== 顶层 await (仅 ESM) ==========
// config.mjs
const response = await fetch('https://api.example.com/config');
export const config = await response.json();
// 其他模块导入此文件时会等待此 Promise 完成
```

## 性能与最佳实践

- **静态分析**：ESM 的 import 必须在顶层，编译器可以在不执行代码的情况下分析依赖关系
- **Tree Shaking**：ESM 支持静态分析消除未使用的导出，减小打包体积
- **`node:` 协议**：使用 `import fs from 'node:fs'` 明确标识 Node.js 内置模块，避免与 npm 包混淆
- **文件扩展名**：ESM 中导入相对路径必须包含扩展名（`.js`, `.mjs`），不像 CJS 可以省略

<!-- Converted from: 3_Node-ESM模块.html -->
