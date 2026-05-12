# 选项式 API 生命周期

## 一、概念说明
Vue 2/3 选项式 API 使用**选项对象**的方式定义生命周期钩子。每个钩子是一个与特定名称匹配的方法。了解选项式 API 的生命周期有助于维护旧项目和理解 Vue 的演进。

## 二、具体用法

### 2.1 完整的选项式 API 生命周期
```vue
<script>
export default {
  beforeCreate() {
    console.log('beforeCreate: 实例初始化，data/methods 还不可用')
  },
  created() {
    console.log('created: 实例创建完成，data/methods 可用')
  },
  beforeMount() {
    console.log('beforeMount: 模板编译完成，DOM 未挂载')
  },
  mounted() {
    console.log('mounted: DOM 已挂载')
  },
  beforeUpdate() {
    console.log('beforeUpdate: 数据变化，DOM 未更新')
  },
  updated() {
    console.log('updated: DOM 已更新')
  },
  beforeUnmount() {
    console.log('beforeUnmount: 即将卸载')
  },
  unmounted() {
    console.log('unmounted: 已卸载')
  },
  activated() {
    console.log('activated: keep-alive 组件激活')
  },
  deactivated() {
    console.log('deactivated: keep-alive 组件失活')
  },
  errorCaptured(err, instance, info) {
    console.error('errorCaptured:', err)
    return false
  }
}
</script>
```

### 2.2 选项式 API 中访问数据
```vue
<script>
export default {
  data() {
    return { count: 0, items: [] }
  },
  async mounted() {
    // this 访问组件实例
    console.log(this.count)
    this.items = await fetch('/api/items').then(r => r.json())
  }
}
</script>
```

## 三、注意事项与常见陷阱
- `beforeCreate` 和 `created` 在组合式 API 中等价于 `setup()` 函数本身
- `beforeDestroy` 和 `destroyed` 在 Vue 3 中已重命名为 `beforeUnmount`/`unmounted`
- 不要混用选项式和组合式 API 的生命周期钩子
- 选项式 API 仍然完全支持，但新项目推荐使用组合式 API

## 四、选项式 API 中的 this 上下文

```vue
<script>
export default {
  data() {
    return { count: 0, items: [] }
  },
  computed: {
    doubleCount() { return this.count * 2 }
  },
  methods: {
    increment() { this.count++ },
    async loadItems() {
      this.items = await fetch('/api/items').then(r => r.json())
    }
  },
  watch: {
    count(newVal, oldVal) {
      console.log(`count 从 ${oldVal} 变为 ${newVal}`)
    }
  },
  created() {
    console.log('count:', this.count)
  },
  mounted() {
    this.$refs.myElement.focus()
  },
  beforeUnmount() {
    clearInterval(this.timer)
  }
}
</script>
```

## 五、Vue 2 vs Vue 3 命名变化

| Vue 2 | Vue 3 | 说明 |
| --- | --- | --- |
| beforeCreate | beforeCreate | 未变 |
| created | created | 未变 |
| beforeMount | beforeMount | 未变 |
| mounted | mounted | 未变 |
| beforeUpdate | beforeUpdate | 未变 |
| updated | updated | 未变 |
| beforeDestroy | **beforeUnmount** | 重命名 |
| destroyed | **unmounted** | 重命名 |
| activated | activated | 未变 |
| deactivated | deactivated | 未变 |
| errorCaptured | errorCaptured | 未变 |

## 六、注意事项补充

- Vue 3 中 `beforeDestroy` 和 `destroyed` 仍然可用但已废弃
- 选项式 API 的钩子在 `<script setup>` 中不可用
- 不要在选项式 API 的 methods 中使用箭头函数，`this` 不会指向组件实例
- mixin 的钩子与组件钩子合并执行
