# 类型导入import-type

## 一、概念说明

`import type` 语法用于只导入类型，编译时会被完全移除。在 `isolatedModules` 模式下是必需的，也优化了编译输出的大小。

## 二、具体用法

### 2.1 基本语法

```typescript
// 只导入类型
import type { User, CreateUserDto } from './types';
import type { Response } from 'express';

// 混合导入 — 值和类型
import { UserService } from './services/user';
import type { User } from './types';

// 导入整个模块的类型
import type * as Types from './types';
```

### 2.2 类型导入内联语法

```typescript
// TypeScript 4.5+ 支持内联类型导入
import { type User, UserService } from './services';

// 等价于
import type { User } from './services';
import { UserService } from './services';
```

### 2.3 verbatimModuleSyntax

```typescript
// tsconfig.json
{
  "compilerOptions": {
    "verbatimModuleSyntax": true // 强制使用 import type
  }
}

// 使用后必须区分
import type { User } from './types'; // 正确
import { User } from './types';       // 如果 User 只有类型，报错
```

### 2.4 导出类型

```typescript
// 导出类型
export type { User, CreateUserDto } from './types';

// 重新导出
export type { default as UserService } from './services/user';

// 导出所有类型
export type * from './types';
```

### 2.5 常量枚举与类型导入

```typescript
// const enum 值需要值导入
import { Direction } from './enums'; // 值导入

// 普通 enum 类型可以类型导入
import type { Status } from './enums'; // 类型导入
```

## 三、注意事项与常见陷阱

1. **`import type` 不会出现在编译输出中**：减小 bundle 大小
2. **`isolatedModules` 强制使用 `import type`**
3. **`verbatimModuleSyntax` 更严格**：推荐在新项目中使用
4. **内联类型导入更简洁**：`import { type X, Y }`
5. **值和类型混合导入时区分清楚**
