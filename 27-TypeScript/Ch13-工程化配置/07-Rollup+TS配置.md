# Rollup+TS配置

## 一、概念说明

Rollup 是库打包的首选工具，支持通过 `@rollup/plugin-typescript` 处理 TypeScript。Rollup 擅长生成干净的 ESM/CJS 双格式输出，适合发布 npm 包。

## 二、具体用法

### 2.1 基本配置

```bash
npm install -D rollup @rollup/plugin-typescript @rollup/plugin-node-resolve @rollup/plugin-commonjs
```

```typescript
// rollup.config.mjs
import typescript from '@rollup/plugin-typescript';
import resolve from '@rollup/plugin-node-resolve';
import commonjs from '@rollup/plugin-commonjs';
import dts from 'rollup-plugin-dts';

export default [
  // 编译 JS
  {
    input: 'src/index.ts',
    output: [
      { file: 'dist/index.cjs', format: 'cjs' },
      { file: 'dist/index.mjs', format: 'esm' },
    ],
    plugins: [
      resolve(),
      commonjs(),
      typescript({ tsconfig: './tsconfig.json' }),
    ],
    external: ['react', 'vue'],
  },
  // 生成类型声明
  {
    input: 'dist/index.d.ts',
    output: [{ file: 'dist/index.d.ts', format: 'esm' }],
    plugins: [dts()],
  },
];
```

### 2.2 tsconfig 配置

```json
{
  "compilerOptions": {
    "target": "ES2020",
    "module": "ESNext",
    "declaration": true,
    "declarationDir": "./dist",
    "outDir": "./dist",
    "strict": true,
    "esModuleInterop": true
  },
  "include": ["src/**/*"]
}
```

### 2.3 多入口配置

```typescript
export default {
  input: {
    index: 'src/index.ts',
    utils: 'src/utils.ts',
    types: 'src/types.ts',
  },
  output: {
    dir: 'dist',
    format: 'esm',
    preserveModules: true, // 保留模块结构
  },
  plugins: [typescript()],
};
```

### 2.4 package.json

```json
{
  "type": "module",
  "main": "dist/index.cjs",
  "module": "dist/index.mjs",
  "types": "dist/index.d.ts",
  "exports": {
    ".": {
      "types": "./dist/index.d.ts",
      "import": "./dist/index.mjs",
      "require": "./dist/index.cjs"
    }
  },
  "files": ["dist"],
  "scripts": {
    "build": "rollup -c"
  }
}
```

## 三、注意事项与常见陷阱

1. **`external` 必须排除 peer dependencies**：不要把依赖打包进去
2. **`preserveModules: true` 保留目录结构**：支持 tree-shaking
3. **类型声明用 `rollup-plugin-dts` 合并**：生成单个 `.d.ts` 文件
4. **Rollup 不支持 HMR**：开发时用 Vite
5. **CommonJS 模块需要 `@rollup/plugin-commonjs`**
