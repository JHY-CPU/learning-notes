# ESLint 配置

## 一、概念说明

ESLint 是 JavaScript/TypeScript 的代码检查工具。配合 `@typescript-eslint` 插件，可以在编写 TypeScript 时检测代码风格问题和潜在错误。TypeScript 5.x 推荐使用 ESLint 的 flat config 格式（`eslint.config.js`）。

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

## 三、注意事项与常见陷阱

1. **ESLint 不替代编译器**：ESLint 检查风格和常见错误，类型检查仍需 `tsc`
2. **与 Prettier 配合**：用 `eslint-config-prettier` 禁用 ESLint 的格式化规则
3. **性能优化**：大项目使用 `--cache` 选项加速重复检查
4. **规则继承**：`tseslint.configs.recommended` 包含了 TypeScript 推荐规则
