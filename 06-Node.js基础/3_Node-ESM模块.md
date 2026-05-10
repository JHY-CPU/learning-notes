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


<!-- Converted from: 3_Node-ESM模块.html -->
