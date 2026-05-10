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


<!-- Converted from: 1_Node简介与环境.html -->
