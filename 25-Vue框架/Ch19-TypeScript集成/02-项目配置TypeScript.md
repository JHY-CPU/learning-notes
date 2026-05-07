# 项目配置TypeScript

## 一、概念说明

配置 Vue 3 + TypeScript 项目需要 `tsconfig.json` 和 Vite 的 TS 支持。Vite 使用 `esbuild` 进行 TS 转译（不进行类型检查），类型检查由 IDE 或 `vue-tsc` 命令完成。`tsconfig.json` 中的 `vue-ts` 插件让 TS 识别 `.vue` 文件。

## 二、具体用法

### Vite 创建 TS 项目

```bash
# 创建 Vue 3 + TypeScript 项目
npm create vite@latest my-vue-app -- --template vue-ts
cd my-vue-app
npm install

# 项目结构：
# my-vue-app/
# ├── src/
# │   ├── App.vue
# │   ├── main.ts          ← TS 入口
# │   ├── components/
# │   ├── vite-env.d.ts    ← Vite 类型声明
# ├── tsconfig.json
# ├── tsconfig.node.json
# └── vite.config.ts
```

### tsconfig.json 配置

```json
{
  "compilerOptions": {
    "target": "ES2020",
    "module": "ESNext",
    "moduleResolution": "bundler",
    "strict": true,              // 启用严格模式
    "jsx": "preserve",
    "resolveJsonModule": true,
    "isolatedModules": true,
    "esModuleInterop": true,
    "lib": ["ES2020", "DOM", "DOM.Iterable"],
    "skipLibCheck": true,
    "noEmit": true,              // Vite 负责编译，TS 只做检查
    "paths": {
      "@/*": ["./src/*"]         // 路径别名
    },
    "types": ["vite/client"]
  },
  "include": [
    "src/**/*.ts",
    "src/**/*.d.ts",
    "src/**/*.tsx",
    "src/**/*.vue"               // 包含 .vue 文件
  ]
}
```

### Vite 配置路径别名

```ts
// vite.config.ts
import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import { resolve } from 'path'

export default defineConfig({
  plugins: [vue()],
  resolve: {
    alias: {
      '@': resolve(__dirname, 'src')
    }
  }
})
```

### 类型检查命令

```json
// package.json
{
  "scripts": {
    "dev": "vite",
    "build": "vue-tsc --noEmit && vite build",
    // vue-tsc：检查 Vue 和 TS 类型
    // --noEmit：只检查不输出文件
    "preview": "vite preview",
    "type-check": "vue-tsc --noEmit"
  }
}
```

### Vite 类型声明

```ts
// src/vite-env.d.ts
/// <reference types="vite/client" />

// 声明 .vue 模块
declare module '*.vue' {
  import type { DefineComponent } from 'vue'
  const component: DefineComponent<{}, {}, any>
  export default component
}

// 声明静态资源
declare module '*.svg' {
  const src: string
  export default src
}
```

## 三、注意事项与常见陷阱

1. **Vite 不做类型检查**：`vite build` 不会因为类型错误而失败，需在 CI 中运行 `vue-tsc`
2. **strict: true 是推荐配置**：禁用严格模式会失去大部分 TS 优势
3. **路径别名需要两处配置**：tsconfig.json 和 vite.config.ts 都要配置 `@`
4. **tsconfig.node.json 单独管理**：Node.js 相关配置（如 vite.config.ts）放在单独文件
5. **IDE 需要重启**：修改 tsconfig.json 后需要重启 IDE 或 TS 服务生效
