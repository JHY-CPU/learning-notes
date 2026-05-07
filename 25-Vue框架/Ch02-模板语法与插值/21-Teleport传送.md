# Teleport 传送

## 一、概念说明

`<Teleport>` 是 Vue 3 内置组件，允许将子组件渲染到 DOM 中的其他位置，而非父组件的 DOM 树中。常用于模态框、通知、弹出层等需要脱离当前 DOM 层级的场景。

```vue
<script setup>
import { ref } from 'vue'

const showModal = ref(false)
</script>

<template>
  <div class="container">
    <h1>应用内容</h1>
    <button @click="showModal = true">打开模态框</button>

    <!-- 将模态框传送到 body 下 -->
    <Teleport to="body">
      <div v-if="showModal" class="modal">
        <h2>模态框</h2>
        <p>我被渲染在 body 元素下，而非当前组件位置</p>
        <button @click="showModal = false">关闭</button>
      </div>
    </Teleport>
  </div>
</template>

<style scoped>
.modal {
  position: fixed;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  background: white;
  padding: 20px;
  border-radius: 8px;
  box-shadow: 0 2px 10px rgba(0,0,0,0.3);
}
</style>
```

## 二、具体用法

### 2.1 传送到指定目标

```vue
<template>
  <!-- 传送到 body -->
  <Teleport to="body">
    <div class="tooltip">提示内容</div>
  </Teleport>

  <!-- 传送到指定元素 -->
  <Teleport to="#modal-container">
    <MyModal />
  </Teleport>

  <!-- 动态目标 -->
  <Teleport :to="targetElement">
    <Notification />
  </Teleport>
</template>
```

### 2.2 禁用 Teleport

```vue
<template>
  <!-- 条件性禁用传送 -->
  <Teleport to="body" :disabled="isMobile">
    <!-- 在移动端不传送，直接在原位置渲染 -->
    <Sidebar />
  </Teleport>
</template>
```

### 2.3 多个 Teleport 到同一目标

```vue
<template>
  <Teleport to="body">
    <ModalA />
  </Teleport>

  <Teleport to="body">
    <ModalB />
    <!-- 多个 Teleport 的内容按顺序追加 -->
  </Teleport>
</template>
```

## 三、注意事项与常见陷阱

- Teleport 的 CSS 样式来自组件本身，不受目标容器影响
- Teleport 只影响 DOM 位置，不影响组件的父子关系
- Vue DevTools 能正确展示 Teleport 的组件层级
- SSR 使用 Teleport 时需要特殊处理
- 目标元素必须在 Teleport 组件挂载前已存在于 DOM 中
