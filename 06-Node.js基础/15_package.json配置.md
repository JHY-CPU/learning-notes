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


## 高级配置字段

```json
{
    "name": "my-package",
    "version": "1.0.0",

    "type": "module",
    "exports": {
        ".": {
            "import": "./dist/index.mjs",
            "require": "./dist/index.cjs",
            "types": "./dist/index.d.ts"
        },
        "./utils": "./dist/utils.js",
        "./package.json": "./package.json"
    },
    "main": "./dist/index.cjs",
    "module": "./dist/index.mjs",
    "types": "./dist/index.d.ts",

    "workspaces": [
        "packages/*"
    ],

    "scripts": {
        "start": "node index.js",
        "dev": "nodemon --watch src index.js",
        "build": "tsc && node scripts/build.js",
        "test": "jest --coverage",
        "lint": "eslint . --ext .js,.ts",
        "format": "prettier --write 'src/**/*.{js,ts}'",
        "prepublishOnly": "npm run build && npm test",
        "prepare": "husky install",
        "postinstall": "node scripts/postinstall.js"
    },

    "husky": {
        "hooks": {
            "pre-commit": "lint-staged",
            "commit-msg": "commitlint -E HUSKY_GIT_MSG"
        }
    },

    "lint-staged": {
        "*.{js,ts}": ["eslint --fix", "prettier --write"],
        "*.{json,md}": ["prettier --write"]
    },

    "browserslist": [
        "> 0.5%",
        "last 2 versions",
        "not dead"
    ],

    "sideEffects": false
}
```

## 生命周期脚本

```bash
# ========== npm 脚本生命周期 ==========
# 按以下顺序自动执行:

# 安装时:
# preinstall → install → postinstall → prepublish → prepare

# 发布时:
# prepublish → prepack → prepare → postpack → publish → postpublish

# 常用钩子:
preinstall    # install 前执行
postinstall   # install 后执行 (如编译原生模块)
prepublish    # publish 前执行
prepare       # install 后和 publish 前执行
pretest       # test 前执行
posttest      # test 后执行
prestart      # start 前执行
poststart     # start 后执行
```

## monorepo 配置

```json
// ========== 根目录 package.json ==========
{
    "name": "my-monorepo",
    "private": true,
    "workspaces": [
        "packages/*",
        "apps/*"
    ],
    "scripts": {
        "build": "npm run build --workspaces",
        "test": "npm run test --workspaces",
        "lint": "npm run lint --workspaces"
    },
    "devDependencies": {
        "typescript": "^5.0.0",
        "jest": "^29.0.0"
    }
}

// packages/utils/package.json
{
    "name": "@myorg/utils",
    "version": "1.0.0",
    "main": "dist/index.js"
}

// packages/core/package.json
{
    "name": "@myorg/core",
    "version": "1.0.0",
    "dependencies": {
        "@myorg/utils": "*"   // 引用 workspace 内的包
    }
}
```

## 最佳实践

- **`"private": true"`**：防止意外发布项目到 npm
- **锁定版本**：dependencies 使用 `^`，关键依赖使用精确版本
- **`"files"` 字段**：只发布必要文件，减小包体积
- **`"engines"` 字段**：指定兼容的 Node.js 版本范围
- **条件导出**：使用 `"exports"` 替代 `"main"`，支持双模块格式（CJS + ESM）

<!-- Converted from: 15_package.json配置.html -->
