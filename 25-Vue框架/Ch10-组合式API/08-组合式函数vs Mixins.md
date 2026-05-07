# 组合式函数 vs Mixins

## 一、概念说明

Mixins是选项式API中复用逻辑的方式，但存在命名冲突、来源不明、难以复用值等问题。组合式函数是Vue 3推荐的替代方案。

```vue
<!-- 旧方式：Mixins -->
<script>
import mouseMixin from './mixins/mouse'
export default {
  mixins: [mouseMixin],
  // x, y 来自mixin，但不知道来源
}
</script>

<!-- 新方式：组合式函数 -->
<script setup>
import { useMouse } from './composables/useMouse'
const { x, y } = useMouse() // 明确知道来源
</script>
```

## 二、具体用法

### Mixins的问题

```js
// mixins/logger.js - 问题示例
export default {
  data() {
    return { count: 0 } // 可能与其他mixin或组件data冲突
  },
  methods: {
    log(msg) { console.log(msg) } // 来源不明确
  },
  mounted() {
    this.log('组件已挂载') // this指向不清晰
  }
}
```

### 组合式函数的优势

```js
// composables/useLogger.js - 对比
export function useLogger(prefix = '') {
  const log = (msg) => console.log(`[${prefix}] ${msg}`)
  const error = (msg) => console.error(`[${prefix}] ${msg}`)

  onMounted(() => log('组件已挂载'))

  return { log, error }
}
```

### 对比总结

| 特性 | Mixins | 组合式函数 |
|------|--------|------------|
| 命名冲突 | 会 | 不会（独立作用域） |
| 数据来源 | 不清晰 | 清晰（返回值解构） |
| 值复用 | 困难 | 容易（直接调用函数） |
| TypeScript | 不友好 | 优秀支持 |
| 逻辑组织 | 分散在各选项 | 集中在一个函数 |
| 推荐度 | 不推荐 | 推荐 |

## 三、注意事项与常见陷阱

1. Mixins中同名选项的合并规则复杂（data会递归合并，生命周期会依次调用）
2. 组合式函数每次调用创建独立实例，天然避免冲突
3. 组合式函数可以接收参数，实现更灵活的配置
4. 从Mixins迁移时，将每个mixin改为一个组合式函数
5. 组合式函数的返回值可以继续在其他组合式函数中使用
