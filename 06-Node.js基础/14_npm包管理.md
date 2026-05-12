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


## npm 实用技巧

```bash
# ========== 包信息查看 ==========
$ npm view express                    # 查看包信息
$ npm view express versions           # 查看所有版本
$ npm view express dependencies       # 查看依赖
$ npm view express@4.18.0             # 查看指定版本

# ========== 依赖分析 ==========
$ npm list                            # 查看已安装依赖树
$ npm list --depth=0                  # 只看顶层依赖
$ npm list lodash                     # 查看指定包版本
$ npm audit                           # 安全漏洞检查
$ npm audit fix                       # 自动修复漏洞

# ========== 清理与重装 ==========
$ npm cache clean --force             # 清理缓存
$ rm -rf node_modules package-lock.json && npm install  # 完全重装

# ========== npx (临时执行) ==========
$ npx create-react-app my-app         # 临时下载并执行
$ npx http-server                     # 不安装直接用
$ npx eslint src/                     # 使用项目本地版本

# ========== 发布包 ==========
$ npm login                           # 登录 npm
$ npm publish                         # 发布包
$ npm publish --access public         # 发布 scoped 包
$ npm unpublish my-package@1.0.0      # 撤销发布 (72h内)
```

## 包管理器对比

| 特性 | npm | yarn | pnpm |
|------|-----|------|------|
| 安装速度 | 中等 | 快 | 最快 |
| 磁盘占用 | 大 | 中等 | 小 (硬链接) |
| 幽灵依赖 | 有 | 有 | 无 (严格模式) |
| lock 文件 | package-lock.json | yarn.lock | pnpm-lock.yaml |
| workspace | 支持 | 支持 | 支持 (原生) |
| 推荐场景 | 通用 | 旧项目 | 新项目、monorepo |

```bash
# ========== pnpm 常用命令 ==========
$ pnpm install                        # 安装依赖
$ pnpm add express                    # 添加依赖
$ pnpm add -D jest                    # 添加开发依赖
$ pnpm remove express                 # 移除依赖
$ pnpm run dev                        # 运行脚本

# ========== yarn 常用命令 ==========
$ yarn install                        # 安装依赖
$ yarn add express                    # 添加依赖
$ yarn add --dev jest                 # 添加开发依赖
$ yarn remove express                 # 移除依赖
```

## 依赖安全最佳实践

```javascript
// ========== .npmrc 配置 ==========
// registry=https://registry.npmmirror.com  (国内镜像)
// save-exact=true                          锁定精确版本
// engine-strict=true                       强制引擎版本

// ========== 依赖版本策略 ==========
// dependencies:  使用 ^ 允许 minor 更新
// devDependencies: 使用 ^ 或 ~
// 关键依赖: 使用精确版本 "1.2.3"
// 定期运行 npm outdated 检查更新

// ========== 供应链安全 ==========
// 1. 使用 package-lock.json 锁定版本
// 2. 定期 npm audit 检查漏洞
// 3. 使用 npm ci 替代 npm install (CI 环境)
// 4. 考虑使用 Snyk 或 Socket 持续监控
```

<!-- Converted from: 14_npm包管理.html -->
