# CommonJS模块


## CommonJS 模块


require、module.exports/exports、循环依赖、模块缓存。


## CommonJS 语法


```
// ========== 导出 ==========
// math.js
const PI = 3.14159;
function sum(a, b) { return a + b; }

// 方式 1: module.exports
module.exports = { PI, sum };

// 方式 2: exports 快捷方式
exports.PI = PI;
exports.sum = sum;

// ⚠️ 不能直接给 exports 赋值:
// exports = { PI, sum }; // ❌ 断开引用

// ========== 导入 ==========
// app.js
const math = require('./math.js');
console.log(math.PI);
console.log(math.sum(1, 2));

// 解构导入
const { PI, sum } = require('./math.js');

// ========== 模块缓存 ==========
// require 会缓存模块，第二次加载使用缓存
const m1 = require('./math.js'); // 加载并执行
const m2 = require('./math.js'); // 从缓存取 (不执行)

// 清除缓存 (调试用)
delete require.cache[require.resolve('./math.js')];

// ========== 加载顺序 ==========
require('./module');   // 相对路径
require('fs');         // 内置模块
require('lodash');     // node_modules 包
require('/abs/path');  // 绝对路径
```


> **Note:** 💡 CommonJS 是同步加载，适合服务端。
>
>
> 💡 exports 是 module.exports 的引用。
>
>
> 💡 ES 模块使用 import/export，与 CJS 不同。
>
>
> ⚠️ 循环依赖需谨慎处理。


## 演示：模块系统

点击按钮查看


<!-- Converted from: 2_CommonJS模块.html -->
