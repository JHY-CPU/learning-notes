# 与 JavaScript 共存

## 一、概念说明

TypeScript 支持渐进式迁移策略，允许在项目中同时使用 `.ts` 和 `.js` 文件。通过 `allowJs` 编译选项和 JSDoc 类型注解，可以逐步将 JavaScript 代码迁移到 TypeScript，而不需要一次性重写所有代码。这在大型遗留项目中非常实用，团队可以按模块逐步迁移。

## 二、具体用法

### 2.1 allowJs 配置

```json
// tsconfig.json
{
  "compilerOptions": {
    "allowJs": true,
    "checkJs": true,
    "outDir": "./dist",
    "strict": false
  },
  "include": ["src/**/*"]
}
```

### 2.2 JSDoc 类型注解

```javascript
// src/utils.js - 在 JS 文件中使用 JSDoc 添加类型

/**
 * 计算两数之和
 * @param {number} a
 * @param {number} b
 * @returns {number}
 */
function add(a, b) {
  return a + b;
}

/** @type {string} */
const message = "Hello from JS";

console.log(add(10, 20));
console.log(message);
// 输出:
// 30
// Hello from JS
```

### 2.3 渐进迁移流程

```bash
# 第一步：项目中已有 JS 文件
src/
├── index.js
├── utils.js
└── api.js

# 第二步：添加 tsconfig.json，开启 allowJs + checkJs

# 第三步：将文件逐个重命名为 .ts
mv src/utils.js src/utils.ts

# 第四步：在 .ts 文件中添加类型注解
# 第五步：重复直到所有文件迁移完成
```

### 2.4 在 TS 中导入 JS 模块

```typescript
// src/index.ts
// 可以直接导入 .js 文件（TypeScript 会查找同名 .ts）
import { add } from "./utils.js";

const result = add(5, 3);
console.log(`结果: ${result}`);
// 输出: 结果: 8
```

### 2.5 JSDoc 复杂类型示例

```javascript
// src/api.js —— 使用 JSDoc 描述复杂类型

/**
 * @typedef {Object} User
 * @property {number} id
 * @property {string} name
 * @property {string} email
 */

/**
 * 获取用户信息
 * @param {number} id
 * @returns {Promise<User>}
 */
async function getUser(id) {
  const response = await fetch(`/api/users/${id}`);
  return response.json();
}

/**
 * @template T
 * @param {T[]} items
 * @param {(item: T) => boolean} predicate
 * @returns {T[]}
 */
function filterItems(items, predicate) {
  return items.filter(predicate);
}
```

### 2.6 混合项目中的导入导出

```typescript
// src/types.ts —— TypeScript 文件定义类型
export interface Config {
  apiUrl: string;
  timeout: number;
}

// src/index.ts —— 导入 JS 函数和 TS 类型
import { fetchData } from "./api.js";     // JS 模块
import type { Config } from "./types.js"; // TS 类型

const config: Config = {
  apiUrl: "https://api.example.com",
  timeout: 5000,
};

fetchData(config.apiUrl);
```

## 三、注意事项与常见陷阱

1. **`checkJs` 选项**：开启后 TypeScript 也会检查 `.js` 文件的类型错误，初期可能报大量错误
2. **模块解析**：导入时仍写 `.js` 扩展名，TypeScript 会自动解析到 `.ts` 或 `.js` 文件
3. **JSDoc 的局限**：复杂类型（泛型、联合类型）用 JSDoc 表达困难，应尽早迁移到 `.ts`
4. **不要用 `@ts-ignore`**：它是逃避类型检查的手段，应优先修复类型问题
5. **循序渐进**：先从工具函数和纯逻辑模块开始迁移，UI 组件和框架代码最后处理
6. **类型声明补充**：对于无法迁移的第三方 JS 代码，编写 `.d.ts` 声明文件补充类型
