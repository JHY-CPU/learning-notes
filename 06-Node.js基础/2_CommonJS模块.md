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


## 循环依赖详解

```javascript
// ========== 循环依赖示例 ==========
// a.js
const b = require('./b');
console.log('a 中 b 的值:', b.value);
exports.done = 'a 完成';

// b.js
const a = require('./a');
console.log('b 中 a 的值:', a.done); // undefined! (a 还未导出)
exports.value = 'b 的值';

// 运行 a.js 输出:
// b 中 a 的值: undefined    ← 因为 a.js 还在加载中
// a 中 b 的值: b 的值

// ========== 解决方案 ==========
// 方案1: 将共享逻辑提取到第三个模块
// shared.js — a.js 和 b.js 都依赖 shared.js

// 方案2: 延迟加载 (在函数内 require)
exports.getValue = function() {
    const b = require('./b'); // 此时 b 已完全加载
    return b.value;
};
```

## 模块缓存与热更新

```javascript
// ========== 模块缓存机制 ==========
// require.cache 是一个对象，key 为模块绝对路径
console.log(Object.keys(require.cache)); // 已加载的模块列表

// 清除单个模块缓存
function invalidateModule(modulePath) {
    const resolved = require.resolve(modulePath);
    delete require.cache[resolved];
}

// 清除模块及其所有依赖的缓存
function invalidateWithDeps(modulePath) {
    const resolved = require.resolve(modulePath);
    const module = require.cache[resolved];
    if (module) {
        // 递归清除引用了该模块的父模块
        module.children.forEach(child => {
            invalidateWithDeps(child.id);
        });
        delete require.cache[resolved];
    }
}

// ========== 实现简易热更新 ==========
function hotRequire(modulePath) {
    invalidateModule(modulePath);
    return require(modulePath);
}
// ⚠️ 生产环境不推荐手动管理缓存
```

## module.exports vs exports 陷阱

```javascript
// ========== exports 是 module.exports 的引用 ==========
// 初始时: exports === module.exports (都是 {})

// ✅ 正确 — 添加属性
exports.foo = 1;
exports.bar = 2;

// ✅ 正确 — 替换 module.exports
module.exports = { foo: 1, bar: 2 };

// ❌ 错误 — 直接赋值给 exports 会断开引用
exports = { foo: 1, bar: 2 }; // 失效！require 得到的是 module.exports

// ========== 类导出 ==========
// math.js
class Calculator {
    add(a, b) { return a + b; }
}
module.exports = Calculator;

// app.js
const Calculator = require('./math');
const calc = new Calculator();
console.log(calc.add(1, 2)); // 3
```

<!-- Converted from: 2_CommonJS模块.html -->
