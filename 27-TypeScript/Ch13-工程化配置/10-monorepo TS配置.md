# monorepo TS配置

## 一、概念说明

Monorepo 中的 TypeScript 配置需要处理多包的类型共享、项目引用和构建顺序。Turborepo、pnpm workspaces 是常用的 monorepo 工具，配合 TS 项目引用使用效果最佳。

## 二、具体用法

### 2.1 pnpm workspace 配置

```yaml
# pnpm-workspace.yaml
packages:
  - 'packages/*'
  - 'apps/*'
```

### 2.2 根 tsconfig

```json
// tsconfig.json（根目录）
{
  "compilerOptions": {
    "strict": true,
    "target": "ES2022",
    "module": "ESNext",
    "moduleResolution": "bundler",
    "skipLibCheck": true,
    "isolatedModules": true,
    "noEmit": true
  },
  "files": [],
  "references": [
    { "path": "./packages/shared" },
    { "path": "./packages/ui" },
    { "path": "./apps/web" },
    { "path": "./apps/api" }
  ]
}
```

### 2.3 共享包配置

```json
// packages/shared/tsconfig.json
{
  "compilerOptions": {
    "composite": true,
    "declaration": true,
    "declarationMap": true,
    "outDir": "./dist",
    "rootDir": "./src"
  },
  "include": ["src/**/*"]
}
```

### 2.4 应用包配置

```json
// apps/web/tsconfig.json
{
  "compilerOptions": {
    "composite": true,
    "declaration": true,
    "outDir": "./dist",
    "jsx": "react-jsx"
  },
  "references": [
    { "path": "../../packages/shared" },
    { "path": "../../packages/ui" }
  ],
  "include": ["src/**/*"]
}
```

### 2.5 Turborepo 配置

```json
// turbo.json
{
  "tasks": {
    "build": {
      "dependsOn": ["^build"],
      "outputs": ["dist/**"]
    },
    "type-check": {
      "dependsOn": ["^build"],
      "outputs": []
    },
    "dev": {
      "cache": false,
      "persistent": true
    }
  }
}
```

### 2.6 包间引用

```typescript
// packages/shared/src/index.ts
export type { User, CreateUserDto } from './types/user';
export { formatDate, formatCurrency } from './utils/format';

// apps/web/src/App.tsx
import { User, formatDate } from '@myorg/shared'; // 跨包引用
```

## 三、注意事项与常见陷阱

1. **共享包必须用 `composite: true`**：启用项目引用
2. **`references` 定义构建顺序**：确保依赖先构建
3. **路径别名在 monorepo 中容易混乱**：用包名代替路径别名
4. **Turborepo 的 `dependsOn: ["^build"]`**：先构建依赖包
5. **每个包的 `package.json` 要正确配置 `exports`**
