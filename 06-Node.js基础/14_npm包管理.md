# npm包管理


## npm 包管理


npm init / install / uninstall、语义版本、锁文件、scripts 脚本。


## npm 命令大全


```
// ========== 初始化 ==========
$ npm init                          // 交互式创建 package.json
$ npm init -y                       // 默认值创建
$ npm init -y --scope @mycompany    // 带 scope

// ========== 安装包 ==========
$ npm install express                // 安装到 dependencies
$ npm i express                      // 简写
$ npm install -D jest                // 安装到 devDependencies
$ npm install -g typescript          // 全局安装
$ npm install express@4.18.0         // 指定版本
$ npm install express@latest         // 最新版

// ========== 卸载/更新 ==========
$ npm uninstall express              // 卸载
$ npm update                        // 更新所有包
$ npm update express                 // 更新单个
$ npm outdated                       // 检查过期包

// ========== 语义版本 ==========
// semver: major.minor.patch
// ^1.2.3 — 兼容 minor (>=1.2.3 <2.0.0)
// ~1.2.3 — 兼容 patch (>=1.2.3 <1.3.0)
// 1.2.3  — 精确版本
// *      — 任意版本
// >=1.2.3 <2.0.0 — 范围

// ========== 锁文件 ==========
// package-lock.json — 锁定依赖版本
// 必须提交到 git
// 确保所有环境使用相同版本

// ========== scripts ==========
// package.json scripts
// {
//   "scripts": {
//     "start": "node server.js",
//     "dev": "nodemon server.js",
//     "test": "jest",
//     "build": "webpack --mode production",
//     "lint": "eslint ."
//   }
// }
// $ npm run dev
```


## 演示：npm 功能

点击按钮查看


<!-- Converted from: 14_npm包管理.html -->
