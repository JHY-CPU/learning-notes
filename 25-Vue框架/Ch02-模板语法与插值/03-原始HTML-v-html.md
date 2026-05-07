# 原始 HTML - v-html

## 一、概念说明

`v-html` 指令用于输出真正的 HTML 内容。当数据包含 HTML 标签时，`{{ }}` 会将标签转义为纯文本，而 `v-html` 会将 HTML 直接插入到 DOM 中。

**警告**：`v-html` 存在 XSS（跨站脚本攻击）风险，永远不要用它渲染用户提交的内容。

```vue
<script setup>
import { ref } from 'vue'

const rawHtml = ref('<strong>加粗文本</strong> 和 <em>斜体文本</em>')
</script>

<template>
  <!-- 转义输出（安全） -->
  <p>{{ rawHtml }}</p>
  <!-- 显示: <strong>加粗文本</strong> 和 <em>斜体文本</em> -->

  <!-- HTML 输出（危险） -->
  <p v-html="rawHtml"></p>
  <!-- 显示: 加粗文本 和 斜体文本 -->
</template>
```

## 二、具体用法

### 2.1 渲染富文本内容

```vue
<script setup>
import { ref } from 'vue'

const articleContent = ref(`
  <h2>文章标题</h2>
  <p>这是一段<strong>富文本</strong>内容。</p>
  <ul>
    <li>列表项 1</li>
    <li>列表项 2</li>
  </ul>
`)
</script>

<template>
  <div class="article" v-html="articleContent"></div>
</template>
```

### 2.2 安全使用 v-html

```vue
<script setup>
import { ref } from 'vue'
import DOMPurify from 'dompurify'

// 使用 DOMPurify 过滤危险内容
const unsafeHtml = ref('<script>alert("xss")</script><p>安全内容</p>')
const safeHtml = computed(() => DOMPurify.sanitize(unsafeHtml.value))
</script>

<template>
  <div v-html="safeHtml"></div>
</template>
```

## 三、注意事项与常见陷阱

- **永远不要**用 `v-html` 渲染用户输入的内容（存在 XSS 漏洞）
- `v-html` 会替换元素的所有子内容，不会与 `{{ }}` 混用
- `scoped` 样式对 `v-html` 插入的内容不生效，需要 `:deep()` 选择器
- SVG 命名空间中的 `v-html` 不受支持
- 使用第三方库（如 DOMPurify）过滤不安全的 HTML
