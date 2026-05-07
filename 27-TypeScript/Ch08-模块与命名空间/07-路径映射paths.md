# 路径映射 paths

## 一、概念说明

`paths` 在 `tsconfig.json` 中定义模块路径别名，避免冗长的相对路径。常用于设置 `@/` 别名指向 `src/` 目录。

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
      "@utils/*": ["src/utils/*"]
    }
  }
}
```

### 2.2 使用别名

```typescript
// 不使用别名
import { User } from "../../../types/user";
import { Button } from "../../../components/Button";

// 使用别名（更简洁）
import { User } from "@/types/user";
import { Button } from "@components/Button";

const user: User = { id: 1, name: "Alice" };
console.log(user.name);
// 输出: Alice
```

### 2.3 构建工具配置

```javascript
// vite.config.ts - Vite 需要配套配置
import { defineConfig } from "vite";
import path from "path";

export default defineConfig({
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "src"),
    },
  },
});
```

## 三、注意事项与常见陷阱

1. **`baseUrl` 必须设置**：`paths` 相对于 `baseUrl` 解析
2. **构建工具需配套配置**：`paths` 只影响 TypeScript，构建工具需单独配置
3. **运行时不解析**：需要 `tsconfig-paths` 等工具在运行时解析
4. **不要过度嵌套**：2-3 个别名即可，过多反而混乱
