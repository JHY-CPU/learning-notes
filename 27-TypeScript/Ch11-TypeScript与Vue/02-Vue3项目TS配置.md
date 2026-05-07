# Vue3项目TS配置

## 一、概念说明

Vue 3 + TypeScript 项目的配置相对简单，Vite 原生支持 Vue 和 TypeScript。关键配置包括 tsconfig.json、Vite 配置和类型声明文件。

## 二、具体用法

### 2.1 创建项目

```bash
npm create vite@latest my-vue-app -- --template vue-ts
cd my-vue-app
npm install
```

### 2.2 tsconfig.json

```json
{
  "compilerOptions": {
    "target": "ES2020",
    "module": "ESNext",
    "moduleResolution": "bundler",
    "strict": true,
    "jsx": "preserve",
    "resolveJsonModule": true,
    "isolatedModules": true,
    "esModuleInterop": true,
    "lib": ["ES2020", "DOM", "DOM.Iterable"],
    "skipLibCheck": true,
    "noEmit": true,
    "baseUrl": ".",
    "paths": {
      "@/*": ["src/*"]
    },
    "types": ["vite/client"]
  },
  "include": ["src/**/*.ts", "src/**/*.d.ts", "src/**/*.tsx", "src/**/*.vue"],
  "references": [{ "path": "./tsconfig.node.json" }]
}
```

### 2.3 tsconfig.node.json

```json
{
  "compilerOptions": {
    "composite": true,
    "module": "ESNext",
    "moduleResolution": "bundler",
    "allowSyntheticDefaultImports": true,
    "strict": true
  },
  "include": ["vite.config.ts"]
}
```

### 2.4 Vite 配置

```typescript
// vite.config.ts
import { defineConfig } from 'vite';
import vue from '@vitejs/plugin-vue';
import { fileURLToPath, URL } from 'node:url';

export default defineConfig({
  plugins: [vue()],
  resolve: {
    alias: {
      '@': fileURLToPath(new URL('./src', import.meta.url)),
    },
  },
});
```

### 2.5 类型声明文件

```typescript
// src/env.d.ts
/// <reference types="vite/client" />

// Vue 文件的类型声明
declare module '*.vue' {
  import type { DefineComponent } from 'vue';
  const component: DefineComponent<{}, {}, any>;
  export default component;
}

// 环境变量类型
interface ImportMetaEnv {
  readonly VITE_API_URL: string;
  readonly VITE_APP_TITLE: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
```

### 2.6 Volar 配置

```json
// .vscode/settings.json
{
  "typescript.tsdk": "node_modules/typescript/lib",
  "typescript.enablePromptUseWorkspaceTsdk": true,
  "vue.inlayHints.ignoreDirectives": true
}
```

## 三、注意事项与常见陷阱

1. **使用 Volar 插件（非 Vetur）**：Volar 是 Vue 3 的官方 VS Code 插件
2. **`noEmit: true`**：Vue 项目不需要 tsc 输出，只做类型检查
3. **`moduleResolution: "bundler"`**：匹配 Vite 的模块解析行为
4. **`.vue` 文件需要类型声明**：否则 TypeScript 不认识 `.vue` 导入
5. **使用 `vue-tsc` 做类型检查**：`tsc` 不理解 `.vue` 文件
