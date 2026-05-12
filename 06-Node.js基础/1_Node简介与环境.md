# Node简介与环境


## Node.js 简介与环境


Node.js 运行时、版本管理、REPL、运行方式、浏览器环境差异。


## Node.js 概述


```
// ========== Node.js 是什么 ==========
// Node.js 是 JS 的服务器端运行环境
// 基于 Chrome V8 引擎
// 非阻塞 I/O + 事件驱动架构

// ========== 版本管理 ==========
// nvm (Node Version Manager)
//   nvm install 18       # 安装 Node 18
//   nvm use 18           # 切换版本
//   nvm ls               # 查看版本列表

// n (npm 包)
//   n 18.0.0

// ========== REPL (Read-Eval-Print Loop) ==========
// 在终端输入 node 进入交互模式
// $ node
// > console.log('hello')
// > .help     # 查看 REPL 命令
// > .exit     # 退出

// ========== 运行方式 ==========
// $ node script.js           # 运行文件
// $ node --watch script.js   # 热重载 (Node 18+)
// $ node -e "console.log(1)" # 内联代码

// ========== Node vs 浏览器 ==========
// Node.js 特有:
//   - global 全局对象
//   - process, Buffer, __dirname
//   - require/module.exports (CommonJS)
//   - fs, path, http 等内置模块
//   - 没有 DOM, window, document
//   - 没有 Web API (fetch 在 Node 18+ 可用)

// 浏览器特有:
//   - window, document, DOM API
//   - WebSocket, localStorage, IndexedDB
//   - Canvas, WebGL
//   - fetch, XMLHttpRequest
```


## 演示：Node.js 概念

点击按钮查看


## Node.js 事件循环机制

```javascript
// ========== 事件循环阶段 ==========
// 1. timers        — setTimeout/setInterval 回调
// 2. pending       — 系统级回调 (如 TCP 错误)
// 3. poll          — I/O 回调 (大部分回调在此执行)
// 4. check         — setImmediate 回调
// 5. close         — socket.on('close') 等

// ========== setTimeout vs setImmediate ==========
setTimeout(() => console.log('timeout'), 0);
setImmediate(() => console.log('immediate'));
// 顺序不确定！取决于事件循环阶段
// 在 I/O 回调中 setImmediate 一定先执行

const fs = require('fs');
fs.readFile(__filename, () => {
    setTimeout(() => console.log('timeout'), 0);
    setImmediate(() => console.log('immediate')); // 一定先输出
});

// ========== nextTick vs microtask ==========
process.nextTick(() => console.log('nextTick'));
Promise.resolve().then(() => console.log('promise'));
// nextTick 优先级高于 Promise
```

## Node.js 模块系统概览

```javascript
// ========== 内置模块分类 ==========
// 文件系统: fs, path
// 网络: http, https, net, dgram, dns
// 流: stream
// 加密: crypto
// 进程: process, child_process, cluster, worker_threads
// 工具: util, events, buffer, url, querystring
// 调试: console, assert, debugger

// ========== 模块查找机制 ==========
// 1. 核心模块 (fs, http) → 直接加载
// 2. 相对路径 (./module) → 解析为绝对路径
// 3. 第三方模块 (lodash) → node_modules 查找
//    逐级向上查找 node_modules 目录

// ========== require.resolve ==========
// 查找模块路径但不加载
const lodashPath = require.resolve('lodash');
console.log(lodashPath); // 完整路径
```

## 最佳实践

- **版本选择**：LTS 版本（如 18.x, 20.x）适合生产环境，Current 版本（最新特性）适合开发体验
- **环境隔离**：使用 `.nvmrc` 文件统一团队 Node 版本
- **ES模块**：新项目推荐使用 ESM（`"type": "module"`），享受 tree-shaking 和静态分析
- **全局安装谨慎**：只在 CLI 工具时使用 `-g`，项目依赖应安装到本地
- **性能监控**：使用 `node --prof` 生成性能分析数据，或使用 `--inspect` 结合 Chrome DevTools

<!-- Converted from: 1_Node简介与环境.html -->
