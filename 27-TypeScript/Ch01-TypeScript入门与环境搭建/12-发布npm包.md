# 发布 npm 包

## 一、概念说明

发布 TypeScript 编写的 npm 包时，需要正确配置编译输出，确保同时发布 CommonJS 和 ESM 格式的代码，以及对应的类型声明文件（`.d.ts`），让使用者获得完整的类型支持。

## 二、具体用法

### 2.1 tsconfig 配置

```json
// tsconfig.json - 用于开发
{
  "compilerOptions": {
    "target": "ES2020",
    "module": "ESNext",
    "declaration": true,
    "declarationDir": "./dist/types",
    "outDir": "./dist",
    "strict": true
  },
  "include": ["src"]
}
```

### 2.2 package.json 配置

```json
{
  "name": "my-ts-library",
  "version": "1.0.0",
  "main": "./dist/index.js",
  "types": "./dist/types/index.d.ts",
  "exports": {
    ".": {
      "import": "./dist/index.mjs",
      "require": "./dist/index.js",
      "types": "./dist/types/index.d.ts"
    }
  },
  "files": ["dist"],
  "scripts": {
    "build": "tsc",
    "prepublishOnly": "npm run build"
  }
}
```

### 2.3 源码示例

```typescript
// src/index.ts
export function greet(name: string): string {
  return `Hello, ${name}!`;
}

export interface Options {
  uppercase?: boolean;
}

export function format(text: string, options: Options = {}): string {
  return options.uppercase ? text.toUpperCase() : text;
}
```

```bash
# 构建并发布
npm run build
npm publish
```

**使用者获得的类型提示：**
```typescript
// 用户代码
import { greet, format } from "my-ts-library";

greet("World");           // 返回类型: string
format("hello", { uppercase: true }); // 返回类型: string
// 输出: Hello
```

## 三、注意事项与常见陷阱

1. **`declaration: true` 必须开启**：否则用户没有类型提示
2. **`files` 字段**：只发布 `dist` 目录，避免发布源码和配置文件
3. **双格式输出**：同时支持 ESM 和 CJS 需要额外构建工具（如 `tsup`、`unbuild`）
4. **`typesVersions` 兼容**：老版本 Node 需要额外配置 `typesVersions`
