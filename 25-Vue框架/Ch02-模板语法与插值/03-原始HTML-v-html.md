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
  <!-- 显示: <strong>加粗文本</strong>... -->

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

### 2.2 安全使用 v-html（DOMPurify）

```vue
<script setup>
import { ref, computed } from 'vue'
import DOMPurify from 'dompurify'

const unsafeHtml = ref('<script>alert("xss")<\/script><p>安全内容</p>')
const safeHtml = computed(() => DOMPurify.sanitize(unsafeHtml.value))
</script>

<template>
  <div v-html="safeHtml"></div>
</template>
```

```bash
# 安装 DOMPurify
pnpm add dompurify
```

### 2.3 允许特定标签

```vue
<script setup>
import { computed } from 'vue'
import DOMPurify from 'dompurify'

const html = ref('<p>段落</p><a href="javascript:alert(1)">危险链接</a>')

const sanitized = computed(() =>
  DOMPurify.sanitize(html.value, {
    ALLOWED_TAGS: ['p', 'strong', 'em', 'ul', 'li', 'a'],
    ALLOWED_ATTR: ['href', 'title']
  })
)
</script>
```

### 2.4 样式处理

```vue
<style scoped>
/* scoped 样式对 v-html 内容不生效 */
/* 需要使用 :deep() */
.article :deep(h2) {
  color: #42b883;
  border-bottom: 2px solid #42b883;
}

.article :deep(ul) {
  padding-left: 20px;
}

.article :deep(strong) {
  font-weight: 700;
}
</style>
```

## 三、常见用例

| 场景 | 是否推荐 v-html |
|------|---------------|
| 渲染 CMS 富文本内容 | 是（需净化） |
| 渲染 Markdown 转 HTML | 是（需净化） |
| 显示用户评论 | 否（用 `{{ }}` 转义） |
| 显示用户昵称 | 否（用 `{{ }}` 转义） |
| 显示后端返回的 HTML 邮件模板 | 是（需净化） |

## 四、注意事项与常见陷阱

- **永远不要**用 `v-html` 渲染用户输入的内容（存在 XSS 漏洞）
- `v-html` 会替换元素的所有子内容，不会与 `{{ }}` 混用
- `scoped` 样式对 `v-html` 插入的内容不生效，需要 `:deep()` 选择器
- SVG 命名空间中的 `v-html` 不受支持
- 使用第三方库（如 DOMPurify）过滤不安全的 HTML
- `v-html` 不会编译模板语法（如 `{{ }}`、`v-if` 等在 HTML 字符串中无效）
- 后端返回的 HTML 也应经过净化处理
