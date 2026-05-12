# ESLint 配置

## 一、概念说明

ESLint 是 JavaScript/TypeScript 的代码检查工具。配合 `@typescript-eslint` 插件，可以在编写 TypeScript 时检测代码风格问题和潜在错误。TypeScript 5.x 推荐使用 ESLint 的 flat config 格式（`eslint.config.js`）。ESLint 不替代 TypeScript 编译器的类型检查，而是专注于代码风格、潜在 bug 和最佳实践。

## 二、具体用法

### 2.1 安装依赖

```bash
# 安装 ESLint 和 TypeScript 解析器
npm install -D eslint @eslint/js \
  typescript-eslint \
  typescript
```

### 2.2 Flat Config 配置

```javascript
// eslint.config.js
import eslint from "@eslint/js";
import tseslint from "typescript-eslint";

export default tseslint.config(
  eslint.configs.recommended,
  ...tseslint.configs.recommended,
  {
    rules: {
      "@typescript-eslint/no-unused-vars": "warn",
      "@typescript-eslint/explicit-function-return-type": "off",
      "@typescript-eslint/no-explicit-any": "warn",
    },
  }
);
```

### 2.3 运行检查

```bash
# 检查代码
npx eslint src/

# 自动修复
npx eslint src/ --fix
```

### 2.4 常用规则示例

```typescript
// ❌ no-explicit-any: 避免使用 any
// function log(data: any) { console.log(data); }

// ✅ 使用具体类型或 unknown
function log(data: unknown) {
  console.log(data);
}

// ❌ no-unused-vars: 未使用的变量
// const unused = "this will warn";

// ✅ 删除或使用该变量
const used = "this is used";
console.log(used);
// 输出: this is used
```

### 2.5 与 Prettier 配合使用

```bash
# 安装 Prettier 及 ESLint 兼容插件
npm install -D prettier eslint-config-prettier
```

```javascript
// eslint.config.js
import eslint from "@eslint/js";
import tseslint from "typescript-eslint";
import prettier from "eslint-config-prettier";

export default tseslint.config(
  eslint.configs.recommended,
  ...tseslint.configs.recommended,
  prettier, // 关闭与 Prettier 冲突的规则
  {
    rules: {
      "@typescript-eslint/no-explicit-any": "warn",
    },
  }
);
```

### 2.6 自定义规则配置

```javascript
// eslint.config.js —— 更详细的规则配置
export default tseslint.config({
  rules: {
    // 强制使用 === 而非 ==
    eqeqeq: ["error", "always"],

    // 禁止使用 var
    "no-var": "error",

    // 强制使用 const 声明不会重新赋值的变量
    "prefer-const": "error",

    // TypeScript 特有规则
    "@typescript-eslint/no-non-null-assertion": "warn", // 警告 ! 非空断言
    "@typescript-eslint/consistent-type-imports": "error", // 强制使用 import type
    "@typescript-eslint/no-floating-promises": "error", // 必须处理 Promise
  },
});
```

## 三、注意事项与常见陷阱

1. **ESLint 不替代编译器**：ESLint 检查风格和常见错误，类型检查仍需 `tsc`
2. **与 Prettier 配合**：用 `eslint-config-prettier` 禁用 ESLint 的格式化规则，避免冲突
3. **性能优化**：大项目使用 `--cache` 选项加速重复检查
4. **规则继承**：`tseslint.configs.recommended` 包含了 TypeScript 推荐规则
5. **旧配置迁移**：`.eslintrc` 已废弃，应迁移到 `eslint.config.js` flat config 格式
6. **IDE 集成**：安装 VS Code ESLint 插件，可在编辑器中实时看到错误提示
