# Prettier集成

## 一、概念说明

Prettier 是代码格式化工具，与 ESLint 配合使用（ESLint 负责代码质量，Prettier 负责代码格式）。TypeScript 项目中需要正确配置 Prettier 以处理 TS 特有的语法。

## 二、具体用法

### 2.1 安装

```bash
npm install -D prettier eslint-config-prettier
```

### 2.2 配置文件

```json
// .prettierrc
{
  "semi": true,
  "singleQuote": true,
  "tabWidth": 2,
  "trailingComma": "all",
  "printWidth": 100,
  "bracketSpacing": true,
  "arrowParens": "always",
  "endOfLine": "lf",
  "typescript.enable": true
}
```

### 2.3 忽略文件

```
# .prettierignore
dist
node_modules
coverage
*.min.js
package-lock.json
```

### 2.4 VS Code 集成

```json
// .vscode/settings.json
{
  "editor.defaultFormatter": "esbenp.prettier-vscode",
  "editor.formatOnSave": true,
  "editor.formatOnPaste": true,
  "[typescript]": {
    "editor.defaultFormatter": "esbenp.prettier-vscode"
  },
  "[typescriptreact]": {
    "editor.defaultFormatter": "esbenp.prettier-vscode"
  }
}
```

### 2.5 与 ESLint 配合

```json
// package.json scripts
{
  "scripts": {
    "format": "prettier --write \"src/**/*.{ts,tsx,json,css}\"",
    "format:check": "prettier --check \"src/**/*.{ts,tsx,json,css}\"",
    "lint": "eslint src --ext .ts,.tsx",
    "lint:fix": "eslint src --ext .ts,.tsx --fix"
  }
}
```

### 2.6 Git Hooks 集成

```bash
npm install -D husky lint-staged
npx husky init
```

```json
// package.json
{
  "lint-staged": {
    "*.{ts,tsx}": ["eslint --fix", "prettier --write"],
    "*.{json,css,md}": ["prettier --write"]
  }
}
```

```bash
# .husky/pre-commit
npx lint-staged
```

## 三、注意事项与常见陷阱

1. **ESLint 和 Prettier 不要同时控制格式**：用 `eslint-config-prettier` 关闭冲突规则
2. **`trailingComma: "all"`**：推荐使用，减少 git diff
3. **`endOfLine: "lf"`**：统一换行符，避免跨平台问题
4. **CI 中用 `--check` 而非 `--write`**：只检查不修改
5. **Prettier 不支持所有 TS 语法**：如类型导入，但通常没问题
