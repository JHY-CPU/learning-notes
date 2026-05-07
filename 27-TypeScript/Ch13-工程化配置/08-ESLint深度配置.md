# ESLint深度配置

## 一、概念说明

`@typescript-eslint` 提供了针对 TypeScript 的 ESLint 规则，包括类型感知规则和代码风格规则。合理的 ESLint 配置能捕获潜在错误并保持代码一致性。

## 二、具体用法

### 2.1 安装

```bash
npm install -D eslint @typescript-eslint/parser @typescript-eslint/eslint-plugin
```

### 2.2 基本配置

```javascript
// eslint.config.js (ESLint 9+ 平铺配置)
import js from '@eslint/js';
import tsPlugin from '@typescript-eslint/eslint-plugin';
import tsParser from '@typescript-eslint/parser';

export default [
  js.configs.recommended,
  {
    files: ['**/*.ts', '**/*.tsx'],
    languageOptions: {
      parser: tsParser,
      parserOptions: {
        project: './tsconfig.json',
      },
    },
    plugins: {
      '@typescript-eslint': tsPlugin,
    },
    rules: {
      // 推荐规则
      ...tsPlugin.configs.recommended.rules,
      ...tsPlugin.configs['recommended-requiring-type-checking'].rules,

      // 自定义规则
      '@typescript-eslint/no-unused-vars': ['error', { argsIgnorePattern: '^_' }],
      '@typescript-eslint/explicit-function-return-type': 'off',
      '@typescript-eslint/no-explicit-any': 'warn',
      '@typescript-eslint/strict-boolean-expressions': 'off',
    },
  },
];
```

### 2.3 常用规则

```javascript
rules: {
  // 禁止未使用变量（_ 开头的参数忽略）
  '@typescript-eslint/no-unused-vars': ['error', { argsIgnorePattern: '^_' }],

  // 禁止空接口
  '@typescript-eslint/no-empty-interface': 'error',

  // 禁止非空断言
  '@typescript-eslint/no-non-null-assertion': 'warn',

  // 要求使用 as const
  '@typescript-eslint/prefer-as-const': 'error',

  // 禁止使用 any
  '@typescript-eslint/no-explicit-any': 'warn',

  // 类型感知规则（需要 parserOptions.project）
  '@typescript-eslint/no-floating-promises': 'error',
  '@typescript-eslint/no-misused-promises': 'error',
  '@typescript-eslint/await-thenable': 'error',
}
```

### 2.4 与 Prettier 集成

```bash
npm install -D eslint-config-prettier
```

```javascript
// eslint.config.js
import prettier from 'eslint-config-prettier';

export default [
  // ...其他配置
  prettier, // 放在最后，关闭与 Prettier 冲突的规则
];
```

### 2.5 忽略文件

```javascript
// eslint.config.js
export default [
  {
    ignores: ['dist/**', 'node_modules/**', 'coverage/**', '**/*.js'],
  },
  // ...
];
```

## 三、注意事项与常见陷阱

1. **类型感知规则需要 `parserOptions.project`**：配置 `tsconfig.json` 路径
2. **ESLint 9 使用平铺配置**：不再支持 `.eslintrc` 格式
3. **Prettier 集成必须放在规则配置之后**
4. **不要同时启用冲突的规则**
5. **`no-explicit-any` 在迁移阶段可以设为 `warn`**
