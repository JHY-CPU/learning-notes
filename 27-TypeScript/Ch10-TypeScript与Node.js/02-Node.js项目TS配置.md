# Node.js项目TS配置

## 一、概念说明

Node.js 项目的 TypeScript 配置需要特别注意模块系统（ESM/CJS）、目标版本和模块解析策略。合理的 `tsconfig.json` 配置能确保代码在 Node.js 环境中正确运行。

## 二、具体用法

### 2.1 推荐的 tsconfig.json

```json
{
  "compilerOptions": {
    "target": "ES2022",
    "module": "NodeNext",
    "moduleResolution": "NodeNext",
    "lib": ["ES2022"],
    "outDir": "./dist",
    "rootDir": "./src",
    "strict": true,
    "esModuleInterop": true,
    "skipLibCheck": true,
    "forceConsistentCasingInFileNames": true,
    "resolveJsonModule": true,
    "declaration": true,
    "declarationMap": true,
    "sourceMap": true,
    "isolatedModules": true
  },
  "include": ["src/**/*"],
  "exclude": ["node_modules", "dist"]
}
```

### 2.2 ESM 项目配置

```json
{
  "compilerOptions": {
    "module": "NodeNext",
    "moduleResolution": "NodeNext",
    "target": "ES2022"
  }
}
```

```typescript
// ESM 中需要 .js 扩展名（即使源文件是 .ts）
import { createUser } from './services/user.js'; // 注意 .js 后缀
import config from './config.json' with { type: 'json' };
```

### 2.3 CJS 项目配置

```json
{
  "compilerOptions": {
    "module": "CommonJS",
    "moduleResolution": "Node",
    "target": "ES2020"
  }
}
```

### 2.4 项目引用（Monorepo）

```json
// tsconfig.json（根配置）
{
  "references": [
    { "path": "./packages/shared" },
    { "path": "./packages/api" },
    { "path": "./packages/web" }
  ],
  "files": []
}

// packages/shared/tsconfig.json
{
  "compilerOptions": {
    "composite": true,
    "outDir": "./dist",
    "declaration": true
  },
  "include": ["src/**/*"]
}
```

### 2.5 类型声明文件

```typescript
// src/types/env.d.ts — 环境变量类型
declare namespace NodeJS {
  interface ProcessEnv {
    NODE_ENV: 'development' | 'production' | 'test';
    PORT: string;
    DATABASE_URL: string;
    JWT_SECRET: string;
  }
}
```

## 三、注意事项与常见陷阱

1. **ESM 项目必须用 `NodeNext`**：而不是 `ESNext`，`NodeNext` 匹配 Node.js 实际行为
2. **ESM 中导入需要 `.js` 扩展名**：即使实际文件是 `.ts`，编译后对应 `.js`
3. **`esModuleInterop: true`**：让 CJS 模块的默认导入正常工作
4. **`skipLibCheck: true`**：加快编译速度，跳过 `.d.ts` 文件的类型检查
5. **`isolatedModules: true`**：确保代码兼容 Babel/esbuild 等非 TS 编译器
