# 创建React+TS项目

## 一、概念说明

使用 TypeScript 创建 React 项目有多种方式，最主流的是 **Vite**。Vite 提供了极快的开发启动速度和热更新体验，且原生支持 TypeScript，无需额外配置。

CRA（Create React App）虽然经典但已停止维护，不再推荐用于新项目。

## 二、具体用法

### 2.1 使用 Vite 创建（推荐）

```bash
# 创建项目
npm create vite@latest my-app -- --template react-ts

# 安装依赖
cd my-app
npm install

# 启动开发服务器
npm run dev
```

生成的项目结构：
```
my-app/
├── src/
│   ├── App.tsx          # 主组件
│   ├── main.tsx         # 入口文件
│   ├── vite-env.d.ts    # Vite 类型声明
│   └── index.css        # 样式
├── tsconfig.json        # TS 配置
├── tsconfig.app.json    # 应用 TS 配置
├── tsconfig.node.json   # Node TS 配置
├── vite.config.ts       # Vite 配置
└── package.json
```

### 2.2 Vite 配置文件类型化

```typescript
// vite.config.ts — 完整类型化的配置
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path';

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'), // 路径别名
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
});
```

### 2.3 tsconfig.json 关键配置

```json
{
  "compilerOptions": {
    "target": "ES2020",
    "lib": ["ES2020", "DOM", "DOM.Iterable"],
    "module": "ESNext",
    "moduleResolution": "bundler",
    "jsx": "react-jsx",           // 使用新的 JSX 转换
    "strict": true,                // 严格模式
    "noUnusedLocals": true,        // 未使用变量报错
    "noUnusedParameters": true,    // 未使用参数报错
    "baseUrl": ".",
    "paths": {
      "@/*": ["src/*"]             // 路径映射
    }
  },
  "include": ["src"]
}
```

### 2.4 Next.js 创建项目

```bash
npx create-next-app@latest my-next-app --typescript
```

## 三、注意事项与常见陷阱

1. **`react-jsx` 选项**：React 17+ 使用新 JSX 转换，不需要 `import React`
2. **路径别名**：需要同时在 `tsconfig.json` 和 `vite.config.ts` 中配置
3. **`vite-env.d.ts`**：不要删除，它包含了 Vite 特有的类型定义（如 `import.meta.env`）
4. **`strict` 模式**：建议始终开启，虽然前期会有更多报错，但能养成更好的类型习惯
5. **CSS Modules**：Vite 原生支持 `.module.css`，但需要 `*.d.ts` 声明文件配合类型提示
