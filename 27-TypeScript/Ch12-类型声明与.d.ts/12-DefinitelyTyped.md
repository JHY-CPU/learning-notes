# DefinitelyTyped

## 一、概念说明

DefinitelyTyped 是 TypeScript 社区维护的类型声明仓库，包含数千个 JavaScript 库的类型定义。通过 `@types/*` npm 包安装使用，是获取第三方库类型信息的主要方式。

## 二、具体用法

### 2.1 安装类型包

```bash
# 安装单个库的类型
npm install -D @types/node
npm install -D @types/lodash
npm install -D @types/express
npm install -D @types/jest

# 类型包命名规则：
# 库名: lodash      → @types/lodash
# 库名: @scope/pkg  → @types/scope__pkg
# 库名: express     → @types/express
```

### 2.2 tsconfig 配置

```json
{
  "compilerOptions": {
    // 方式一：自动包含所有 @types 包
    "types": [],  // 空数组 = 包含所有

    // 方式二：指定包含哪些
    "types": ["node", "jest"],

    // typeRoots — 自定义类型包搜索路径
    "typeRoots": ["./node_modules/@types", "./custom-types"]
  }
}
```

### 2.3 使用类型包

```typescript
// 安装 @types/lodash 后，lodash 自动有类型
import _ from 'lodash';

const result = _.groupBy([1, 2, 3, 4], n => n % 2);
// result 的类型是 Dictionary<number[]>

// 安装 @types/express 后
import express, { Request, Response } from 'express';

const app = express();
app.get('/', (req: Request, res: Response) => {
  res.json({ message: 'Hello' });
});
```

### 2.4 处理版本不匹配

```typescript
// 某些库自带类型，不需要 @types
// 如 Vue 3、React 18+、Axios 等

// 检查是否需要安装 @types
// 1. 看库的 package.json 中是否有 "types" 或 "typings" 字段
// 2. 看 node_modules 中是否有 .d.ts 文件

// 如果 @types 版本与库版本不匹配
// 指定版本
npm install -D @types/lodash@4.14.191
```

### 2.5 贡献类型声明

```bash
# 为新库添加类型声明
# 1. Fork DefinitelyTyped 仓库
# 2. 在 types/ 目录创建库的类型文件
# 3. 编写类型声明和测试
# 4. 提交 Pull Request

# 本地测试
npm run test-all my-library
npm run lint my-library
```

### 2.6 检查类型包质量

```bash
# 查看类型包的流行度和维护状态
npm info @types/lodash
npm info @types/lodash --json | grep -A5 "time"

# 查看类型包的 GitHub 仓库
# https://github.com/DefinitelyTyped/DefinitelyTyped/tree/master/types/lodash
```

## 三、注意事项与常见陷阱

1. **不是所有库都需要 @types**：现代库自带类型
2. **@types 版本可能落后于库版本**：关注版本兼容
3. **`@types` 包的维护者是社区**：质量参差不齐
4. **全局类型可能冲突**：如 `@types/node` 和 `@types/jest` 有重叠
5. **贡献 DefinitelyTyped 需要遵循规范**：阅读 CONTRIBUTING.md
