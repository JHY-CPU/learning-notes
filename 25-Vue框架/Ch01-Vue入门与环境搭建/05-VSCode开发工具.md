# VS Code 开发工具配置

## 一、概念说明

VS Code 是 Vue 开发的首选编辑器。**Vue - Official**（原 Volar）是 Vue 官方团队维护的 VS Code 插件，提供语法高亮、类型检查、自动补全、错误提示等功能。配合 ESLint 和 Prettier，可以构建完整的代码质量保障体系。

## 二、具体用法

### 2.1 Vue - Official 插件

安装后自动提供：
- `.vue` 文件语法高亮
- 模板中表达式的类型推断
- 组件 props/events 的智能提示
- 模板中跳转到定义

```
安装：Extensions 搜索 "Vue - Official" → Install
```

### 2.2 ESLint 配置

项目中安装 ESLint 并配置 Vue 规则：

```bash
pnpm add -D eslint @vue/eslint-config-prettier
```

```js
// eslint.config.js（ESLint Flat Config）
import pluginVue from 'eslint-plugin-vue'

export default [
  ...pluginVue.configs['flat/essential'],
  {
    rules: {
      'vue/multi-word-component-names': 'off', // 允许单词组件名
      'vue/no-unused-vars': 'warn' // 未使用变量警告
    }
  }
]
```

### 2.3 Prettier 配置

```json
// .prettierrc
{
  "semi": false,
  "singleQuote": true,
  "tabWidth": 2,
  "trailingComma": "all",
  "printWidth": 80,
  "vueIndentScriptAndStyle": true
}
```

### 2.4 保存时自动格式化

```json
// .vscode/settings.json
{
  "editor.formatOnSave": true,
  "editor.defaultFormatter": "esbenp.prettier-vscode",
  "editor.codeActionsOnSave": {
    "source.fixAll.eslint": "explicit"
  }
}
```

### 2.5 推荐的 VS Code 快捷键

| 快捷键 | 功能 |
|--------|------|
| `Ctrl+P` | 快速打开文件 |
| `Ctrl+Shift+P` | 命令面板 |
| `Ctrl+`` ` | 打开终端 |
| `Alt+Shift+F` | 格式化文档 |
| `F12` | 跳转到定义 |
| `Ctrl+.` | 快速修复 |
| `Ctrl+Shift+V` | 预览 Markdown |

## 三、常见用例

### 3.1 代码片段配置

```json
// .vscode/vue.code-snippets
{
  "Vue 3 SFC": {
    "prefix": "v3",
    "body": [
      "<script setup>",
      "import { ref } from 'vue'",
      "",
      "const ${1:state} = ref(${2:null})",
      "</script>",
      "",
      "<template>",
      "  <div>${3:content}</div>",
      "</template>",
      "",
      "<style scoped>",
      "</style>"
    ],
    "description": "Vue 3 单文件组件模板"
  }
}
```

### 3.2 调试配置

```json
// .vscode/launch.json
{
  "version": "0.2.0",
  "configurations": [
    {
      "type": "chrome",
      "request": "launch",
      "name": "Vue: Chrome",
      "url": "http://localhost:5173",
      "webRoot": "${workspaceFolder}/src",
      "sourceMapPathOverrides": {
        "webpack:///src/*": "${webRoot}/*"
      }
    }
  ]
}
```

## 四、注意事项与常见陷阱

- 必须**禁用 Vetur** 插件，否则与 Volar 冲突导致语法错误
- ESLint 9 使用 Flat Config 格式，与旧版 `.eslintrc` 不同
- Prettier 和 ESLint 规则可能冲突，使用 `@vue/eslint-config-prettier` 解决
- `lang="ts"` 的模板需要 TypeScript 插件支持类型推断
- 建议开启 `Volar: Take Over Mode` 以获得最佳性能
- 大型项目中 Volar 可能较慢，可增加 `typescript.tsserver.maxTsServerMemory`
