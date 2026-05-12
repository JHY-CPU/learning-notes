# v-show 详解

## 一、概念说明
`v-show` 通过切换元素的 CSS `display` 属性来控制**显示和隐藏**。与 v-if 不同，元素始终保留在 DOM 中，只是 CSS 切换。

## 二、具体用法

### 2.1 基本用法
```vue
<template>
  <div v-show="isVisible">显示/隐藏内容</div>
  <!-- 渲染结果: <div style="display: none;">显示/隐藏内容</div> -->
</template>
<script setup>
import { ref } from 'vue'
const isVisible = ref(true)
</script>
```

### 2.2 适合频繁切换
```vue
<template>
  <div v-for="tab in tabs" :key="tab.id">
    <button @click="activeTab = tab.id">{{ tab.label }}</button>
  </div>
  <div v-show="activeTab === 'tab1'">Tab 1 内容</div>
  <div v-show="activeTab === 'tab2'">Tab 2 内容</div>
  <div v-show="activeTab === 'tab3'">Tab 3 内容</div>
</template>
<script setup>
import { ref } from 'vue'
const activeTab = ref('tab1')
const tabs = [
  { id: 'tab1', label: '标签1' },
  { id: 'tab2', label: '标签2' },
  { id: 'tab3', label: '标签3' }
]
</script>
```

### 2.3 v-show vs v-if 对比
```vue
<template>
  <!-- v-show: 元素始终在 DOM，display 切换 -->
  <div v-show="show">内容</div>

  <!-- v-if: 元素从 DOM 移除/添加 -->
  <div v-if="show">内容</div>
</template>
```

| 特性 | v-show | v-if |
|------|--------|------|
| DOM 操作 | 始终渲染 | 条件渲染 |
| 初始开销 | 较高（始终编译）| 较低（惰性）|
| 切换开销 | 低（CSS 切换）| 高（DOM 重建）|
| 适合场景 | 频繁切换 | 很少变化 |
| template | 不支持 | 支持 |

## 三、注意事项与常见陷阱
- v-show **不支持** `<template>` 元素
- v-show **不支持** `v-else`
- 初始渲染开销比 v-if 高（因为始终编译和渲染）
- 频繁切换的场景用 v-show 性能更好

## 四、实际应用场景

### 4.1 Tab 切换
```vue
<template>
  <div class="tabs">
    <div class="tab-headers">
      <button
        v-for="tab in tabs"
        :key="tab.id"
        :class="{ active: activeTab === tab.id }"
        @click="activeTab = tab.id"
      >
        {{ tab.label }}
      </button>
    </div>
    <!-- v-show 保持组件状态，切换成本低 -->
    <div v-show="activeTab === 'profile'">
      <UserProfile />
    </div>
    <div v-show="activeTab === 'settings'">
      <Settings />
    </div>
    <div v-show="activeTab === 'notifications'">
      <Notifications />
    </div>
  </div>
</template>
```

### 4.2 表单验证提示
```vue
<template>
  <input v-model="email" type="email" />
  <!-- 频繁显示/隐藏的验证提示用 v-show -->
  <p v-show="email && !isValidEmail" class="error">邮箱格式错误</p>
  <p v-show="email.length > 50" class="error">邮箱过长</p>
</template>
```

### 4.3 折叠面板
```vue
<template>
  <div class="accordion">
    <button @click="isOpen = !isOpen">
      {{ isOpen ? '收起' : '展开' }}
    </button>
    <!-- 频繁切换用 v-show -->
    <div v-show="isOpen" class="panel">
      <slot />
    </div>
  </div>
</template>
```

## 五、v-show 的 CSS 影响

```vue
<template>
  <!-- v-show 应用 style="display: none" -->
  <div v-show="false">隐藏</div>
  <!-- 渲染结果: <div style="display: none;">隐藏</div> -->

  <!-- 如果元素本身有 display 样式，切换时会恢复原有 display -->
  <span v-show="true" style="display: inline-block">显示</span>
  <!-- v-show=false 时: style="display: none"
       v-show=true 时: style="display: inline-block" -->
</template>
```

## 六、v-show 与 CSS 动画

```vue
<template>
  <!-- v-show 不支持 <Transition> 组件 -->
  <!-- 但可以用 CSS 过渡 -->
  <div class="fade" :class="{ hidden: !show }">内容</div>
</template>
<style>
.fade {
  transition: opacity 0.3s ease;
}
.fade.hidden {
  opacity: 0;
  /* 注意：元素仍在 DOM 中 */
}
</style>
```
