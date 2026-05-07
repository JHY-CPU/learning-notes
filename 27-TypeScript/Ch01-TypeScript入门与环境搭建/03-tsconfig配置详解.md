# tsconfig 配置详解

## 一、概念说明

`tsconfig.json` 是 TypeScript 项目的配置文件，它定义了编译选项和需要编译的文件范围。位于项目根目录的 `tsconfig.json` 会指导 `tsc` 编译器的行为，包括编译目标版本、模块系统、严格程度等。

## 二、具体用法

### 2.1 基本结构

```json
{
  "compilerOptions": {
    "target": "ES2020",
    "module": "ESNext",
    "strict": true,
    "outDir": "./dist",
    "rootDir": "./src"
  },
  "include": ["src/**/*"],
  "exclude": ["node_modules", "dist"]
}
```

### 2.2 常用 compilerOptions

```typescript
// tsconfig.json 中的关键选项说明

// target: 编译输出的 JS 版本
// "ES5" | "ES6" | "ES2020" | "ESNext"

// module: 模块系统
// "CommonJS" | "ESNext" | "AMD" | "UMD"

// strict: 启用所有严格检查（推荐开启）
// 等价于启用 noImplicitAny, strictNullChecks, strictFunctionTypes 等

// outDir: 编译输出目录
// rootDir: 源文件根目录

// 示例：一个典型的生产配置
const config = {
  compilerOptions: {
    target: "ES2020",
    module: "commonjs",
    lib: ["ES2020", "DOM"],
    strict: true,
    outDir: "./dist",
    rootDir: "./src",
    sourceMap: true,
    declaration: true,
   esModuleInterop: true,
    skipLibCheck: true,
    forceConsistentCasingInFileNames: true
  }
};
```

**输出：**
```
配置后执行 tsc 编译，会将 src 目录下的 .ts 文件编译到 dist 目录
```

### 2.3 include 与 exclude

```json
{
  "include": [
    "src/**/*.ts",
    "src/**/*.tsx"
  ],
  "exclude": [
    "node_modules",
    "dist",
    "**/*.spec.ts"
  ],
  "files": [
    "src/index.ts"
  ]
}
```

## 三、注意事项与常见陷阱

1. **`files` vs `include`**：`files` 指定确切文件列表，`include` 支持通配符
2. **`strict` 模式**：新项目务必开启，可避免大量潜在错误
3. **`esModuleInterop`**：使用 CommonJS 模块时建议开启，解决 `import` 兼容性
4. **`skipLibCheck`**：跳过 `.d.ts` 文件检查，加快编译速度
5. **继承配置**：使用 `extends` 可继承其他 `tsconfig.json` 的配置
