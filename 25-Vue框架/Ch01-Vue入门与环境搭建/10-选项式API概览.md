# 选项式 API 概览

## 一、概念说明

选项式 API（Options API）是 Vue 2 以来的经典写法，通过 `data`、`methods`、`computed`、`watch` 等选项组织组件逻辑。Vue 3 完全支持选项式 API，适合从 Vue 2 迁移或偏好该风格的开发者。

```vue
<script>
export default {
  data() {
    return {
      count: 0,
      message: '你好 Vue!'
    }
  },
  methods: {
    increment() {
      this.count++
    }
  },
  computed: {
    doubled() {
      return this.count * 2
    }
  },
  watch: {
    count(newVal) {
      console.log('count变为:', newVal)
    }
  }
}
</script>

<template>
  <p>{{ message }}</p>
  <p>{{ count }} x 2 = {{ doubled }}</p>
  <button @click="increment">+1</button>
</template>
```

## 二、具体用法

### 2.1 完整选项式 API 结构

```vue
<script>
export default {
  name: 'MyComponent',

  // 组件属性
  props: {
    title: { type: String, required: true }
  },

  // 局部注册组件
  components: {},

  // 响应式数据
  data() {
    return { count: 0 }
  },

  // 计算属性（有缓存）
  computed: {
    doubleCount() { return this.count * 2 }
  },

  // 方法
  methods: {
    increment() { this.count++ }
  },

  // 侦听器
  watch: {
    count(newVal, oldVal) {
      console.log(`${oldVal} → ${newVal}`)
    }
  },

  // 生命周期钩子
  mounted() {
    console.log('组件已挂载')
  },

  beforeUnmount() {
    console.log('组件即将卸载')
  }
}
</script>
```

### 2.2 生命周期钩子

```
beforeCreate → created → beforeMount → mounted
→ beforeUpdate → updated → beforeUnmount → unmounted
```

## 三、注意事项与常见陷阱

- `data` 必须是函数，返回对象，避免组件间数据共享
- `this` 在箭头函数中不指向组件实例，methods 中使用普通函数
- 不要在 `data` 或 `computed` 中使用箭头函数
- `watch` 中的 `this` 可以访问组件所有属性
- Vue 3 移除了 `beforeCreate` 和 `created`（在 `setup()` 中等效）
