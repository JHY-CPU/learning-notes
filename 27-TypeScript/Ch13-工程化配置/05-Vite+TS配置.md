# Vite+TS配置

## 一、概念说明

Vite 是现代前端构建工具的首选，原生支持 TypeScript。配置 `vite.config.ts` 可以自定义开发服务器、构建选项和插件。

## 二、具体用法

### 2.1 基本配置

```typescript
// vite.config.ts
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path';

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  server: {
    port: 3000,
    proxy: {
      '/api': {
        target: 'http://localhost:8080',
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: 'dist',
    sourcemap: true,
    minify: 'esbuild',
  },
});
```

### 2.2 库模式

```typescript
export default defineConfig({
  build: {
    lib: {
      entry: 'src/index.ts',
      name: 'MyLib',
      formats: ['es', 'cjs', 'umd'],
      fileName: (format) => `my-lib.${format}.js`,
    },
    rollupOptions: {
      external: ['vue', 'react'],
      output: {
        globals: {
          vue: 'Vue',
          react: 'React',
        },
      },
    },
  },
});
```

### 2.3 环境变量类型

```typescript
// vite-env.d.ts
/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_API_URL: string;
  readonly VITE_APP_TITLE: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
```

### 2.4 插件配置

```typescript
import { defineConfig } from 'vite';
import dts from 'vite-plugin-dts';
import checker from 'vite-plugin-checker';

export default defineConfig({
  plugins: [
    // 自动生成类型声明
    dts({ include: ['src'] }),

    // 开发时类型检查
    checker({
      typescript: true,
      eslint: {
        lintCommand: 'eslint src',
      },
    }),
  ],
});
```

### 2.5 CSS 配置

```typescript
export default defineConfig({
  css: {
    modules: {
      localsConvention: 'camelCaseOnly',
    },
    preprocessorOptions: {
      scss: {
        additionalData: `@import "@/styles/variables.scss";`,
      },
    },
  },
});
```

## 三、注意事项与常见陷阱

1. **`vite.config.ts` 本身被 tsconfig.node.json 管理**：不被应用 tsconfig 覆盖
2. **路径别名需要同时配置 Vite 和 tsconfig**
3. **Vite 使用 esbuild 编译 TS**：不支持 `const enum` 等特性
4. **`import.meta.env` 的类型在 `vite-env.d.ts` 中定义**
5. **生产构建使用 Rollup**：比开发时的 esbuild 更慢但功能更全
