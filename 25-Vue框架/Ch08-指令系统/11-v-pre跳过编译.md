# v-pre 跳过编译

## 一、概念说明
`v-pre` 让 Vue **跳过该元素及其子元素的编译**。元素中的原始 Mustache 语法 `{{ }}` 会被原样显示，不会被替换为数据。适合展示源码或文档。

## 二、具体用法

### 2.1 显示原始语法
```vue
<template>
  <!-- 显示原始的 Mustache 语法 -->
  <span v-pre>{{ 这不会被编译 }}</span>
  <!-- 输出: {{ 这不会被编译 }} -->

  <!-- 没有 v-pre 的话 -->
  <span>{{ 这会被编译 }}</span>
  <!-- 如果没有这个变量，输出空 -->
</template>
```

### 2.2 代码文档展示
```vue
<template>
  <div class="code-example">
    <h3>Vue 模板语法示例</h3>
    <pre v-pre>
&lt;template&gt;
  &lt;p&gt;{{ message }}&lt;/p&gt;
&lt;/template&gt;
    </pre>
  </div>
</template>
```

### 2.3 跳过大量静态内容的编译
```vue
<template>
  <!-- 包含大量不需要 Vue 处理的静态 HTML -->
  <div v-pre>
    <p>这些内容不会被 Vue 编译</p>
    <p>可以加快初始渲染速度</p>
    <p>适用于纯静态内容</p>
  </div>
</template>
```

### 2.4 混合使用
```vue
<template>
  <!-- 只有这部分需要 Vue 处理 -->
  <p>动态内容: {{ dynamicValue }}</p>

  <!-- 这部分保持原样 -->
  <div v-pre>
    <p>模板语法: {{ example }}</p>
    <p>指令: v-if="条件"</p>
  </div>
</template>
<script setup>
import { ref } from 'vue'
const dynamicValue = ref('会更新')
</script>
```

## 三、注意事项与常见陷阱
- v-pre 跳过**整个子树**的编译，包含所有子元素
- 可以减少编译开销，适合纯静态内容区块
- 不要在 v-pre 中放置需要响应式的元素
- 常用于文档网站、技术博客中展示 Vue 代码
