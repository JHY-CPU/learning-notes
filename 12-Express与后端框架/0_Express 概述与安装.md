# Express 概述与安装


## 🚂 Express 概述与安装


Express.js 介绍、安装与项目初始化、Hello World 示例、应用结构、express() 核心方法、Nodemon 开发热重载、常见目录组织。


## Express 概述


```
// ========== Express.js ==========
// Node.js 最流行的 Web 框架
// 基于 Node.js http 模块的封装
// 轻量、灵活、中间件机制

// ========== 核心特性 ==========
// 1. 路由系统 (app.METHOD)
// 2. 中间件 (Middleware)
// 3. 请求/响应处理 (req/res)
// 4. 模板引擎集成
// 5. 静态文件服务

// ========== 安装 ==========
// 初始化项目:
mkdir my-app && cd my-app
npm init -y

// 安装 Express:
npm install express

// 安装开发依赖:
npm install --save-dev nodemon

// package.json 配置:
{
  "scripts": {
    "start": "node index.js",
    "dev": "nodemon index.js"
  }
}

// ========== Hello World ==========
const express = require('express');
const app = express();
const port = 3000;

// 路由定义
app.get('/', (req, res) => {
    res.send('Hello World!');
});

// 启动服务器
app.listen(port, () => {
    console.log(`Server running at http://localhost:${port}`);
});

// ========== 应用结构 ==========
// ┌────────────────────────────────────┐
// │ my-app/                            │
// │ ├── node_modules/                  │
// │ ├── src/                           │
// │ │   ├── routes/    (路由)          │
// │ │   ├── controllers/ (控制器)      │
// │ │   ├── middleware/  (中间件)      │
// │ │   ├── models/     (数据模型)     │
// │ │   ├── services/   (业务逻辑)     │
// │ │   ├── utils/      (工具)         │
// │ │   └── app.js      (入口)         │
// │ ├── public/         (静态文件)     │
// │ ├── views/          (模板)         │
// │ ├── .env            (环境变量)     │
// │ └── package.json                   │
// └────────────────────────────────────┘
```


## 核心方法


```
// ========== express() 核心方法 ==========

// ========== 路由方法 ==========
app.get('/path', handler)       // GET 请求
app.post('/path', handler)      // POST 请求
app.put('/path', handler)       // PUT 请求
app.delete('/path', handler)    // DELETE 请求
app.patch('/path', handler)     // PATCH 请求
app.all('/path', handler)       // 所有 HTTP 方法
app.route('/path')              // 链式定义

// 链式路由:
app.route('/users')
    .get((req, res) => { /* 查询用户列表 */ })
    .post((req, res) => { /* 创建用户 */ })
    .put((req, res) => { /* 更新用户 */ });

// ========== 中间件方法 ==========
app.use(middleware)                    // 全局中间件
app.use('/path', middleware)           // 路径前缀中间件

// 内置中间件:
express.json()                         // JSON 解析
express.urlencoded({ extended: true }) // 表单解析
express.static('public')               // 静态文件

// ========== 启动方法 ==========
app.listen(port, callback)             // 启动 HTTP 服务器
app.listen(port, host, callback)       // 指定主机

// ========== 设置方法 ==========
app.set('view engine', 'ejs')          // 设置模板引擎
app.set('trust proxy', true)           // 信任代理
app.set('env', 'production')           // 设置环境
app.disable('x-powered-by')            // 禁用 X-Powered-By

// ========== Express 生成器 ==========
// 全局安装生成器:
npm install -g express-generator

// 生成项目:
express my-app --view=ejs --git

// 生成器创建的结构:
// ├── bin/www        (启动脚本)
// ├── public/        (静态文件)
// ├── routes/        (路由)
// ├── views/         (模板)
// └── app.js         (主应用)
```


## 中间件概念


```
// ========== 中间件 ==========
// Express 的核心: 请求经过一系列中间件处理

// ┌──────────────────────────────────────────┐
// │  客户端 → 中间件1 → 中间件2 → 路由 → 响应  │
// │                ↓ (错误)                    │
// │            错误处理中间件                   │
// └──────────────────────────────────────────┘

// 中间件函数签名:
function myMiddleware(req, res, next) {
    // req: 请求对象
    // res: 响应对象
    // next: 下一个中间件函数
    next();
}

// 中间件可以:
// 1. 修改 req/res 对象
// 2. 结束请求-响应周期 (res.send/json)
// 3. 调用 next() 传递到下一个中间件

// ========== Nodemon ==========
// 开发时自动重启服务器:
// package.json:
{
  "scripts": {
    "dev": "nodemon src/app.js",
    "start": "node src/app.js"
  }
}

// nodemon.json 配置:
{
  "watch": ["src"],
  "ext": "js,json",
  "ignore": ["node_modules", "test"],
  "delay": "1000"
}

// ========== 最佳实践 ==========
// 1. 使用环境变量 (process.env.PORT)
// 2. 开发用 nodemon
// 3. 目录结构清晰划分
// 4. 中间件顺序重要!
//    常见顺序: 日志→解析→CORS→路由→错误
// 5. 生产用 PM2 管理进程
```


> **Note:** 💡 Express 是最流行的 Node.js Web 框架, 基于中间件机制。核心: app.METHOD(path, handler) 定义路由; app.use() 挂载中间件; express.json() 解析 JSON; listen() 启动。生产注意: 设置 NODE_ENV=production, 使用 PM2 管理进程, 禁用 x-powered-by。


## 练习


<!-- Converted from: 0_Express 概述与安装.html -->
