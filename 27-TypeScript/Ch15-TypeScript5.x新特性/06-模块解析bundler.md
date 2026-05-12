# 模块解析bundler

## 一、概念说明

`moduleResolution: "bundler"` 是 TypeScript 5.0 引入的模块解析策略，模拟 Vite、Webpack 等打包工具的解析行为。它比 `"node"` 更宽松（不需要扩展名），比 `"node16"` 更灵活（不要求 `.js`），支持 `exports` 字段。这是现代前端项目的推荐配置。

## 二、具体用法

### 2.1 配置

```json
// tsconfig.json
{
  "compilerOptions": {
    "module": "ESNext",
    "moduleResolution": "bundler",
    "target": "ES2020"
  }
}
```

### 2.2 与其他策略对比

| 特性 | node | node16 | bundler |
|------|------|--------|---------|
| 需要扩展名 | 否 | 是 (.js) | 否 |
| 支持 `exports` | 部分 | 完整 | 完整 |
| `index` 解析 | 是 | 否 | 是 |
| 支持路径映射 | 是 | 有限 | 是 |
| 适用场景 | CJS | Node ESM | Vite/Webpack |

### 2.3 实际行为

```typescript
// bundler 模式 — 不需要扩展名
import { add } from "./math";           // ✅ 自动解析 .ts/.tsx/.js
import { config } from "./config";      // ✅
import { User } from "@/types/user";    // ✅ 支持 paths 映射

// 支持 package.json 的 exports 字段
import { createApp } from "vue";        // ✅ 正确解析 vue 的 exports
import { z } from "zod";               // ✅
```

### 2.4 配合 verbatimModuleSyntax

```json
{
  "compilerOptions": {
    "module": "ESNext",
    "moduleResolution": "bundler",
    "verbatimModuleSyntax": true
  }
}
```

```typescript
// verbatimModuleSyntax 强制区分值和类型导入
import type { User } from "./types";     // ✅ 编译后消除
import { UserService } from "./service"; // ✅ 编译后保留
```

### 2.5 与构建工具配合

```typescript
// vite.config.ts
import { defineConfig } from "vite";

export default defineConfig({
  resolve: {
    alias: {
      "@": "/src",
    },
  },
});

// TypeScript 的 bundler 策略与 Vite 的解析行为一致
// 不需要额外配置 tsconfig paths 的运行时解析
```

### 2.6 与 JavaScript 的对比

```javascript
// Node.js ESM：必须写扩展名
import { add } from "./math.js";

// Node.js CJS：不需要扩展名
const { add } = require("./math");

// Vite/Webpack：不需要扩展名
import { add } from "./math";

// TypeScript bundler：匹配 Vite/Webpack 行为
import { add } from "./math"; // ✅ 不需要扩展名
```

## 三、注意事项与常见陷阱

1. **推荐用于 Vite/Webpack 项目**：与构建工具的解析行为完全一致
2. **Node.js ESM 项目用 `node16`**：需要 `.js` 扩展名，`bundler` 不适合
3. **`module` 和 `moduleResolution` 必须搭配**：`ESNext` + `bundler` 是标准组合
4. **`exports` 字段中的 `types` 条件**：必须放在最前面
5. **`bundler` 是最宽松的策略**：适合大多数前端项目，不强制约束
6. **TS 5.0+ 特性**：旧版本 TypeScript 不支持此策略
