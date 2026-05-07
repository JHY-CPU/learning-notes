# tsconfig模块解析

## 一、概念说明

`baseUrl`、`paths` 和 `moduleResolution` 控制 TypeScript 如何查找导入的模块。合理的配置可以简化导入路径、支持路径别名和 monorepo 跨包引用。

## 二、具体用法

### 2.1 baseUrl 与 paths

```json
{
  "compilerOptions": {
    "baseUrl": ".",
    "paths": {
      "@/*": ["src/*"],              // @/utils → src/utils
      "@components/*": ["src/components/*"],
      "@shared/*": ["../shared/src/*"],
      "~/*": ["src/*"]               // 另一种别名
    }
  }
}
```

```typescript
// 使用路径别名
import { formatDate } from '@/utils/date';
import { Button } from '@components/Button';
import { User } from '@shared/types';
```

### 2.2 Vite 中配合路径别名

```typescript
// vite.config.ts
import { defineConfig } from 'vite';
import path from 'path';

export default defineConfig({
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
});
```

### 2.3 moduleResolution 选项

```json
{
  "compilerOptions": {
    // "Node" — 经典 Node.js 解析
    // "Node16" / "NodeNext" — 匹配 Node.js ESM/CJS
    // "Bundler" — 匹配 Vite/Webpack 等打包工具
    "moduleResolution": "Bundler"
  }
}
```

### 2.4 解析策略对比

| 策略 | 扩展名解析 | index 解析 | 适用场景 |
|------|-----------|-----------|----------|
| Node | 不需要 | 支持 | CJS 项目 |
| Node16 | 需要 .js | 不支持 | ESM 项目 |
| Bundler | 不需要 | 支持 | Vite/Webpack |

### 2.5 Monorepo 路径

```json
// 根目录 tsconfig
{
  "compilerOptions": {
    "baseUrl": ".",
    "paths": {
      "@myorg/shared": ["packages/shared/src/index.ts"],
      "@myorg/shared/*": ["packages/shared/src/*"]
    }
  }
}
```

## 三、注意事项与常见陷阱

1. **`paths` 需要配合 `baseUrl`**：路径相对于 `baseUrl`
2. **路径别名只在编译时有效**：运行时需要构建工具同步配置
3. **`Bundler` 模式最宽松**：推荐用于 Vite/Webpack 项目
4. **ESM 项目需要 `.js` 扩展名**：即使实际文件是 `.ts`
5. **别名不要用 `$`**：可能与框架冲突
