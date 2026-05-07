# Monorepo管理

## 一、概念说明

Monorepo 将多个相关项目放在一个仓库中管理，共享依赖和工具链。Vue 3 生态推荐 **pnpm workspace** 作为 Monorepo 方案，配合 **Turborepo** 做构建编排。适合组件库、微前端、前后端一体的项目结构。

## 二、具体用法

### pnpm workspace 配置

```yaml
# pnpm-workspace.yaml
packages:
  - 'packages/*'
  - 'apps/*'
```

```json
// package.json - 根项目
{
  "name": "my-monorepo",
  "private": true,
  "scripts": {
    "dev": "turbo run dev",
    "build": "turbo run build",
    "lint": "turbo run lint"
  },
  "devDependencies": {
    "turbo": "^1.10.0"
  }
}
```

### 目录结构

```
my-monorepo/
├── apps/
│   ├── web/              # Vue 3 主应用
│   │   ├── package.json  # { name: "@my/web" }
│   │   └── src/
│   └── admin/            # 管理后台
│       ├── package.json  # { name: "@my/admin" }
│       └── src/
├── packages/
│   ├── ui/               # 共享 UI 组件库
│   │   ├── package.json  # { name: "@my/ui" }
│   │   └── src/
│   ├── utils/            # 共享工具函数
│   │   ├── package.json  # { name: "@my/utils" }
│   │   └── src/
│   └── types/            # 共享类型定义
├── pnpm-workspace.yaml
├── turbo.json
└── package.json
```

### 共享组件库

```ts
// packages/ui/package.json
{
  "name": "@my/ui",
  "version": "1.0.0",
  "main": "./dist/index.js",
  "module": "./dist/index.mjs",
  "types": "./dist/index.d.ts",
  "exports": {
    ".": {
      "import": "./dist/index.mjs",
      "require": "./dist/index.js",
      "types": "./dist/index.d.ts"
    },
    "./style.css": "./dist/style.css"
  }
}
```

```ts
// packages/ui/src/index.ts
export { default as MyButton } from './Button.vue'
export { default as MyCard } from './Card.vue'
```

```vue
<!-- apps/web 中使用 -->
<script setup lang="ts">
import { MyButton, MyCard } from '@my/ui'
import '@my/ui/style.css'
</script>
```

### Turborepo 配置

```json
// turbo.json
{
  "$schema": "https://turbo.build/schema.json",
  "pipeline": {
    "build": {
      "dependsOn": ["^build"],
      "outputs": ["dist/**"]
    },
    "dev": {
      "cache": false,
      "persistent": true
    },
    "lint": {}
  }
}
```

```bash
# 常用命令
pnpm install              # 安装所有依赖
pnpm --filter @my/web dev # 只启动 web 应用
pnpm turbo run build      # 按依赖关系构建所有包
pnpm turbo run build --filter=@my/ui  # 只构建 ui 包
```

## 三、注意事项与常见陷阱

1. **pnpm 的严格依赖隔离**：不同包间不能互相使用未声明的依赖
2. **Turborepo 缓存需配置远程缓存**：CI 中多台机器需要共享缓存
3. **包名用 scope 前缀**：`@my/ui` 避免与 npm 公共包冲突
4. **版本管理用 changesets**：`@changesets/cli` 管理包版本和 CHANGELOG
5. **先构建依赖包**：`dependsOn: ["^build"]` 确保依赖先构建
