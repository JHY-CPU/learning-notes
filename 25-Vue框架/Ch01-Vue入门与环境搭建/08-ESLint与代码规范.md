# ESLint 与代码规范

## 一、概念说明

ESLint 是 JavaScript/TypeScript 的代码检查工具，帮助团队保持一致的代码风格和发现潜在错误。Vue 项目通过 `eslint-plugin-vue` 插件获得 Vue 特有的规则检查。Prettier 负责代码格式化，两者配合形成完整的代码质量体系。

```vue
<script setup>
// ESLint 会检查以下问题：
// 1. 未使用的变量
// 2. 使用 var 而非 let/const
// 3. 缺少分号或多余分号（配合 Prettier）
// 4. 组件命名不规范
import { ref } from 'vue'

const count = ref(0)
</script>

<template>
  <button @click="count++">{{ count }}</button>
</template>
```

## 二、具体用法

### 2.1 安装与配置

```bash
pnpm add -D eslint eslint-plugin-vue @vue/eslint-config-prettier prettier
```

```js
// eslint.config.js（ESLint 9 Flat Config）
import pluginVue from 'eslint-plugin-vue'
import eslintConfigPrettier from '@vue/eslint-config-prettier'

export default [
  {
    name: 'app/files-to-lint',
    files: ['**/*.{js,mjs,jsx,vue}'],
    rules: {
      'vue/multi-word-component-names': 'off',
      'no-unused-vars': 'warn',
      'no-console': 'warn'
    }
  },
  ...pluginVue.configs['flat/essential'],
  eslintConfigPrettier
]
```

### 2.2 npm scripts 配置

```json
// package.json
{
  "scripts": {
    "lint": "eslint . --fix",
    "lint:check": "eslint .",
    "format": "prettier --write src/"
  }
}
```

### 2.3 常用 Vue ESLint 规则

```js
rules: {
  'vue/html-indent': ['error', 2],          // HTML 缩进 2 空格
  'vue/max-attributes-per-line': ['error', {  // 每行最多属性数
    singleline: 3,
    multiline: 1
  }],
  'vue/require-default-prop': 'off',        // 关闭 props 默认值要求
  'vue/singleline-html-element-content-newline': 'off' // 单行元素不换行
}
```

## 三、注意事项与常见陷阱

- ESLint 9 采用 Flat Config，旧版 `.eslintrc` 将被废弃
- Prettier 和 ESLint 规则冲突时，必须安装 `@vue/eslint-config-prettier`
- `--fix` 只能修复可自动修复的规则，语义错误需手动修改
- `.eslintignore` 在 Flat Config 中通过 `ignores` 配置
- 建议在 `pre-commit` 钩子中运行 lint（使用 husky + lint-staged）
