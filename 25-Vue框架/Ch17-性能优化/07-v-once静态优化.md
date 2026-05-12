# v-once 静态优化

## 一、概念说明

`v-once` 指令使元素或组件只渲染一次，后续的数据变化不会触发重新渲染。Vue 会缓存其虚拟 DOM 节点，跳过比对过程。

```vue
<script setup>
import { ref } from 'vue'
const count = ref(0)
const name = ref('Vue')
</script>

<template>
  <!-- 只渲染一次，后续 count 变化不会更新这里 -->
  <div v-once>
    <h1>{{ name }} 学习笔记</h1>
    <p>创建时间: {{ new Date().toLocaleString() }}</p>
  </div>

  <!-- 正常响应式 -->
  <p>计数: {{ count }}</p>
  <button @click="count++">+1</button>
</template>
```

## 二、具体用法

### 2.1 适用场景

```vue
<!-- 静态标题、页脚、版权信息 -->
<footer v-once>
  <p>&copy; 2024 我的网站. 保留所有权利.</p>
</footer>

<!-- 大量静态内容 + 少量动态内容 -->
<div v-once>
  <p>一段很长的静态说明文字...</p>
  <p>这段文字永远不会改变...</p>
</div>

<!-- 静态表单标签 -->
<div v-once>
  <label>用户名</label>
  <label>邮箱</label>
</div>
<input v-model="username" />
```

### 2.2 v-memo 对比（Vue 3.2+）

```vue
<!-- v-once：完全静态，永不更新 -->
<div v-once>{{ staticContent }}</div>

<!-- v-memo：条件缓存，依赖变化才更新 -->
<div v-memo="[id]">
  {{ expensiveComputation(id) }}
</div>
```

### 2.3 包裹整个子树

```vue
<!-- 整个 section 及其子元素都只渲染一次 -->
<section v-once>
  <h2>标题</h2>
  <p>内容</p>
  <ul>
    <li>列表项 1</li>
    <li>列表项 2</li>
  </ul>
</section>
```

## 四、性能收益分析

```vue
<!-- 假设一个有 1000 行静态文本的组件 -->

<!-- 没有 v-once：每次父组件更新都会 diff 1000 个节点 -->
<div>
  <p v-for="i in 1000" :key="i">静态文本 {{ i }}</p>
  <p>动态内容: {{ dynamicValue }}</p>
</div>

<!-- 使用 v-once：1000 个静态节点只 diff 一次 -->
<div v-once>
  <p v-for="i in 1000" :key="i">静态文本 {{ i }}</p>
</div>
<p>动态内容: {{ dynamicValue }}</p>
```

## 五、v-once 与 v-show 对比

| 特性 | v-once | v-show |
|------|--------|--------|
| 首次渲染 | 正常 | 正常 |
| 数据变化 | 不更新 | 更新 |
| DOM 节点 | 保留 | 保留（display:none） |
| 适用场景 | 永不变化的内容 | 频繁切换显示 |
| 性能收益 | 跳过后续 diff | 只跳过 DOM 创建/销毁 |

## 六、实际应用示例

```vue
<template>
  <!-- 静态页脚 -->
  <footer v-once>
    <nav>
      <a href="/about">关于我们</a>
      <a href="/contact">联系方式</a>
      <a href="/privacy">隐私政策</a>
    </nav>
    <p>Copyright 2024</p>
  </footer>

  <!-- 静态表单布局 -->
  <form>
    <div v-once>
      <label>用户名</label>
      <label>邮箱</label>
      <label>密码</label>
    </div>
    <div>
      <input v-model="username" />
      <input v-model="email" />
      <input v-model="password" />
    </div>
  </form>

  <!-- 静态侧边栏内容 -->
  <aside v-once>
    <h3>帮助中心</h3>
    <p>常见问题...</p>
    <p>使用指南...</p>
  </aside>
</template>
```

## 三、注意事项与常见陷阱

- `v-once` 内的所有响应式数据都会失去响应性
- 不要在 `v-once` 内放置用户需要交互的表单元素
- 过度使用 `v-once` 可能导致数据不更新的 bug
- 需要条件性缓存时使用 `v-memo` 更灵活
- `v-once` 的性能收益在大块静态内容时最明显
- 小片段使用 `v-once` 反而增加代码复杂度，不值得
