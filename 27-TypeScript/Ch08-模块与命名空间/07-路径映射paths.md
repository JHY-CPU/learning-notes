# 路径映射 paths

## 一、概念说明

`paths` 在 `tsconfig.json` 中定义模块路径别名，避免冗长的相对路径（如 `../../../utils`）。常用 `@/` 别名指向 `src/` 目录。`paths` 需要 `baseUrl` 作为基准目录，且构建工具也需要配套配置别名解析。

## 二、具体用法

### 2.1 基本路径映射

```json
// tsconfig.json
{
  "compilerOptions": {
    "baseUrl": ".",
    "paths": {
      "@/*": ["src/*"],
      "@components/*": ["src/components/*"],
      "@utils/*": ["src/utils/*"],
      "@types/*": ["src/types/*"]
    }
  }
}
```

### 2.2 使用别名

```typescript
// 不使用别名 — 路径冗长且易出错
import { User } from "../../../types/user";
import { Button } from "../../../components/Button";
import { formatDate } from "../../../utils/date";

// 使用别名 — 简洁清晰
import { User } from "@/types/user";
import { Button } from "@components/Button";
import { formatDate } from "@utils/date";

const user: User = { id: 1, name: "Alice", email: "a@b.com" };
console.log(user.name); // Alice
```

### 2.3 构建工具配套配置

```typescript
// vite.config.ts
import { defineConfig } from "vite";
import path from "path";

export default defineConfig({
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "src"),
      "@components": path.resolve(__dirname, "src/components"),
      "@utils": path.resolve(__dirname, "src/utils"),
    },
  },
});
```

```javascript
// webpack.config.js
module.exports = {
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "src"),
    },
  },
};
```

### 2.4 多组映射

```json
{
  "compilerOptions": {
    "baseUrl": ".",
    "paths": {
      "@app/*": ["src/app/*"],
      "@shared/*": ["src/shared/*"],
      "@assets/*": ["src/assets/*"],
      "#types/*": ["src/types/*"]
    }
  }
}
```

### 2.5 运行时解析

```bash
# ts-node 运行时需要 tsconfig-paths
npm install -D tsconfig-paths

# 使用
npx ts-node -r tsconfig-paths/register src/main.ts

# 或在代码中注册
import "tsconfig-paths/register";
import { main } from "@/app";
```

### 2.6 与 JavaScript 的对比

```javascript
// JavaScript：无内置路径别名，需要构建工具支持
// webpack alias / vite resolve.alias

// TypeScript：tsconfig paths 提供 IDE 类型识别
// 但运行时和构建工具仍需单独配置
// paths 只告诉 TypeScript 编译器如何解析，不修改输出代码
```

## 三、注意事项与常见陷阱

1. **`baseUrl` 必须设置**：`paths` 相对于 `baseUrl` 解析，缺了会报错
2. **构建工具需配套配置**：`paths` 只影响 TypeScript 的类型检查和 IDE，构建工具需单独配 alias
3. **运行时不解析**：Node.js 运行时不认识 `@/`，需要 `tsconfig-paths` 或构建工具处理
4. **别名不宜过多**：2-3 个别名（如 `@/`、`@types/`）即可，过多反而增加认知负担
5. **`*` 通配符**：`@/*` 匹配 `@/` 后面的所有路径，`@utils`（无 `*`）匹配精确路径
6. **`paths` 不影响输出路径**：编译后的 JS 文件路径由 `outDir` 和源文件结构决定
