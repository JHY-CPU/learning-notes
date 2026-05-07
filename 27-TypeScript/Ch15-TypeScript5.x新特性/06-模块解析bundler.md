# 模块解析bundler

## 一、概念说明

`moduleResolution: "bundler"` 是 TypeScript 5.0 引入的模块解析策略，模拟 Vite、Webpack 等打包工具的解析行为。它比 `"Node"` 更宽松，不需要文件扩展名，支持 `exports` 字段。

## 二、具体用法

### 2.1 配置

```json
{
  "compilerOptions": {
    "module": "ESNext",
    "moduleResolution": "bundler"
  }
}
```

### 2.2 与其他策略对比

| 特性 | Node | Node16 | Bundler |
|------|------|--------|---------|
| 需要扩展名 | 否 | 是 (.js) | 否 |
| 支持 `exports` | 部分 | 完整 | 完整 |
| `index` 解析 | 是 | 否 | 是 |
| 适用场景 | CJS | ESM | Vite/Webpack |

### 2.3 使用场景

```typescript
// bundler 模式下 — 不需要扩展名
import { helper } from './utils';
import { User } from '../types';
import config from '@/config';

// 支持 exports 字段
import { createApp } from 'vue'; // 正确解析 vue 的 exports
```

## 三、注意事项与常见陷阱

1. **推荐用于 Vite/Webpack 项目**：匹配打包工具的行为
2. **Node.js ESM 项目用 `Node16`**：需要 `.js` 扩展名
3. **bundler 是最宽松的策略**：适合大多数前端项目
4. **`exports` 字段中的 `types` 条件必须在最前面**
