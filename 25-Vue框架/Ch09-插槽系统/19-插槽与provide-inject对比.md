# 插槽与 Provide-Inject 对比

## 一、概念说明
插槽和 provide/inject 都可以实现**跨组件内容/数据传递**，但它们的设计理念和适用场景完全不同。

## 二、对比分析

### 2.1 本质区别
```
插槽:          内容分发（HTML/模板 → 子组件的占位符）
Provide/Inject: 数据分发（值/函数 → 后代组件的依赖注入）
```

### 2.2 对比表
| 特性 | 插槽 | Provide/Inject |
|------|------|---------------|
| 传递内容 | 模板/DOM | 数据/函数 |
| 作用域 | 父组件 | 祖先组件 |
| 渲染位置 | 子组件插槽位置 | 后代组件任意位置 |
| 响应式 | 通过作用域插槽 | 需手动传 ref |
| 类型安全 | defineSlots | InjectionKey |

### 2.3 插槽适用场景
```vue
<!-- 自定义渲染内容 -->
<List :items="data">
  <template #default="{ item }">
    <span>{{ item.name }}</span>
  </template>
</List>

<!-- 布局组件 -->
<Layout>
  <template #header>标题</template>
  <template #sidebar>导航</template>
</Layout>
```

### 2.4 Provide/Inject 适用场景
```vue
<!-- 注入全局配置 -->
<script setup>
import { provide } from 'vue'
provide('theme', ref('dark'))
provide('locale', ref('zh-CN'))
</script>

<!-- 深层后代使用 -->
<script setup>
const theme = inject('theme')
</script>
```

### 2.5 何时选择哪个
```
需要自定义渲染?     → 插槽
需要传递数据?       → Props / Provide / Pinia
需要传递给深层后代? → Provide/Inject
需要灵活布局?       → 插槽
需要全局配置?       → Provide/Inject
```

## 三、注意事项与常见陷阱
- 插槽和 Provide/Inject 不互斥，可以配合使用
- 不要用 Provide/Inject 做内容分发（应该用插槽）
- 不要用插槽做深层数据传递（应该用 Provide/Inject）
- 选择原则：数据传递用数据方案，内容分发用插槽

## 四、混合使用示例

```vue
<!-- ThemeProvider.vue：用 provide 传递主题数据，用插槽传递内容 -->
<script setup>
import { provide, ref } from 'vue'

const theme = ref('dark')
const toggleTheme = () => {
  theme.value = theme.value === 'dark' ? 'light' : 'dark'
}

provide('theme', theme)
provide('toggleTheme', toggleTheme)
</script>

<template>
  <div :class="`theme-${theme}`">
    <slot :theme="theme" :toggle="toggleTheme" />
  </div>
</template>
```

```vue
<!-- 使用 -->
<template>
  <ThemeProvider v-slot="{ theme, toggle }">
    <!-- 可以通过插槽 props 使用 -->
    <button @click="toggle">当前主题: {{ theme }}</button>
    <!-- 深层组件通过 inject 使用 -->
    <DeepComponent />
  </ThemeProvider>
</template>

<script setup>
const theme = inject('theme')  // 深层组件通过 inject 获取
</script>
```

## 五、Ant Design / Element Plus 的实践

大型 UI 库通常同时使用两者：
- **插槽**：自定义渲染内容（表格列、下拉菜单项）
- **provide/inject**：传递全局配置（语言、尺寸、主题）

```vue
<!-- ElConfigProvider：provide 全局配置 -->
<ElConfigProvider :locale="zhCn" :size="'large'">
  <App />
</ElConfigProvider>

<!-- ElTable：插槽自定义列渲染 -->
<ElTable :data="data">
  <ElTableColumn prop="name" label="姓名">
    <template #default="{ row }">
      <strong>{{ row.name }}</strong>
    </template>
  </ElTableColumn>
</ElTable>
```
