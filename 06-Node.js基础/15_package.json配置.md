# package.json配置


## package.json 配置


name/version/scripts/dependencies/exports/main、字段详解。


## package.json 字段详解


```
// ========== 核心字段 ==========
{
    "name": "@scope/package-name",   // 包名 (可带 scope)
    "version": "1.0.0",              // 语义版本号
    "description": "A useful package",

    // ========== 入口 ==========
    "main": "dist/index.js",         // CJS 入口 (require)
    "module": "dist/index.mjs",      // ESM 入口 (import, 非标准但通用)
    "types": "dist/index.d.ts",      // TypeScript 类型
    "exports": {                     // 条件导出 (Node 12+)
        ".": {
            "import": "./dist/index.mjs",
            "require": "./dist/index.js",
            "types": "./dist/index.d.ts"
        },
        "./utils": "./dist/utils.js"
    },

    // ========== 依赖 ==========
    "dependencies": {},              // 运行时依赖
    "devDependencies": {},           // 开发时依赖
    "peerDependencies": {},          // 宿主依赖 (插件)
    "optionalDependencies": {},      // 可选依赖
    "bundledDependencies": [],       // 打包依赖

    // ========== 脚本 ==========
    "scripts": {
        "start": "node index.js",
        "test": "jest",
        "build": "webpack"
    },

    // ========== 发布 ==========
    "private": true,                 // 私有 (防止发布)
    "files": ["dist/", "README.md"], // 发布的文件
    "license": "MIT",

    // ========== 限制 ==========
    "engines": { "node": ">=18" },   // Node 版本要求
    "os": ["linux", "darwin"],       // 操作系统
    "cpu": ["x64", "arm64"],         // CPU 架构
}
```


## 演示：package.json

点击按钮查看


<!-- Converted from: 15_package.json配置.html -->
