# v-once 一次性渲染

## 一、概念说明
`v-once` 让元素或组件**只渲染一次**。渲染后，即使依赖的响应式数据变化，该部分 DOM 也不会再更新。适用于静态内容的性能优化。

## 二、具体用法

### 2.1 基本用法
```vue
<template>
  <!-- 只渲染一次，后续数据变化不影响 -->
  <p v-once>初始计数: {{ count }}</p>
  <p>当前计数: {{ count }}</p>
  <button @click="count++">+1</button>
  <!-- 点击后："初始计数"不变，"当前计数"更新 -->
</template>
<script setup>
import { ref } from 'vue'
const count = ref(0)
</script>
```

### 2.2 组件级别
```vue
<template>
  <div v-once>
    <Header />
    <Sidebar />
    <p>这些内容不会再更新</p>
  </div>
</template>
```

### 2.3 静态大段内容
```vue
<template>
  <!-- 长文本 / 大量静态 DOM 节省性能 -->
  <div v-once>
    <h1>产品介绍</h1>
    <p>这是一个非常长的静态文本段落...</p>
    <p>不会因为其他数据变化而重新渲染</p>
  </div>

  <!-- 只有这部分会动态更新 -->
  <p>动态数据: {{ dynamicData }}</p>
</template>
<script setup>
import { ref } from 'vue'
const dynamicData = ref('会变化的数据')
</script>
```

### 2.4 在模板中缓存计算结果
```vue
<template>
  <!-- 复杂表达式只计算一次 -->
  <span v-once>{{ expensiveComputation() }}</span>
</template>
```

## 三、注意事项与常见陷阱
- v-once 后的内容将**永远不再更新**，确保确实不需要更新
- 适用于纯静态内容：Logo、页脚、固定文案等
- 与 computed 的区别：computed 是缓存值，v-once 是缓存 DOM
- 不要对需要响应式更新的元素使用 v-once

## 四、性能优化场景

### 4.1 复杂静态组件
```vue
<template>
  <!-- 包含大量 DOM 的静态页脚 -->
  <footer v-once>
    <div class="footer-links">
      <a v-for="link in footerLinks" :key="link.id" :href="link.url">
        {{ link.text }}
      </a>
    </div>
    <div class="copyright">© 2024 Company</div>
  </footer>

  <!-- 其他动态内容正常更新 -->
  <main>{{ dynamicContent }}</main>
</template>
```

### 4.2 大型列表中的静态行
```vue
<template>
  <tr v-for="item in items" :key="item.id">
    <!-- 静态列只渲染一次 -->
    <td v-once>{{ item.createdAt }}</td>
    <td v-once>{{ item.category }}</td>
    <!-- 动态列正常更新 -->
    <td>{{ item.status }}</td>
    <td>{{ item.count }}</td>
  </tr>
</template>
```

## 五、v-once vs v-memo vs computed

| 特性 | v-once | v-memo | computed |
|------|--------|--------|----------|
| 作用对象 | DOM 子树 | DOM 子树 | 数据 |
| 更新 | 永不 | 条件性 | 依赖变化 |
| 适用 | 纯静态 | 大列表优化 | 派生数据 |
| Vue 版本 | 所有 | 3.2+ | 所有 |
