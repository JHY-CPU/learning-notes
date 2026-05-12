# 发布 npm 包

## 一、概念说明

发布 TypeScript 编写的 npm 包时，需要正确配置编译输出，确保同时发布 CommonJS 和 ESM 格式的代码，以及对应的类型声明文件（`.d.ts`），让使用者获得完整的类型支持。现代 npm 包还需要考虑 Tree Shaking 兼容性和多种模块格式的支持。

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

### 2.4 使用 tsup 简化构建

```bash
npm install -D tsup
```

```typescript
// tsup.config.ts
import { defineConfig } from "tsup";

export default defineConfig({
  entry: ["src/index.ts"],
  format: ["cjs", "esm"],   // 同时输出两种格式
  dts: true,                 // 自动生成 .d.ts
  clean: true,               // 构建前清理 dist
  sourcemap: true,
});
```

```json
// package.json
{
  "scripts": {
    "build": "tsup"
  }
}
```

### 2.5 版本管理与发布

```bash
# 更新版本号
npm version patch  # 1.0.0 -> 1.0.1
npm version minor  # 1.0.0 -> 1.1.0
npm version major  # 1.0.0 -> 2.0.0

# 发布到 npm
npm publish

# 发布 beta 版本
npm version prerelease --preid=beta  # 1.0.0 -> 1.0.1-beta.0
npm publish --tag beta
```

## 三、注意事项与常见陷阱

1. **`declaration: true` 必须开启**：否则用户没有类型提示
2. **`files` 字段**：只发布 `dist` 目录，避免发布源码、测试和配置文件
3. **双格式输出**：同时支持 ESM 和 CJS 需要额外构建工具（如 `tsup`、`unbuild`）
4. **`typesVersions` 兼容**：老版本 Node 需要额外配置 `typesVersions`
5. **`sideEffects` 字段**：设置 `"sideEffects": false` 以支持 Tree Shaking
6. **peer dependencies**：框架相关库应将框架声明为 `peerDependencies` 而非 `dependencies`
