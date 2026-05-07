# Props 类型验证

## 一、概念说明

Vue 的 props 支持类型验证，确保父组件传递的数据符合预期。可以指定类型、默认值、是否必填以及自定义验证函数。TypeScript 项目中更推荐使用类型声明。

```vue
<script setup>
defineProps({
  // 类型检查
  age: Number,

  // 多类型
  id: [String, Number],

  // 完整验证对象
  status: {
    type: String,
    required: true,
    validator: (value) => {
      return ['active', 'inactive', 'pending'].includes(value)
    }
  }
})
</script>
```

## 二、具体用法

### 2.1 内置类型验证

```vue
<script setup>
defineProps({
  // 基础类型
  string: String,
  number: Number,
  boolean: Boolean,
  array: Array,
  object: Object,

  // 多种类型
  flexible: [String, Number],

  // 函数类型
  handler: Function
})
</script>
```

### 2.2 带验证的完整声明

```vue
<script setup>
defineProps({
  size: {
    type: String,
    default: 'medium',
    validator: (value) => {
      return ['small', 'medium', 'large'].includes(value)
    }
  },

  count: {
    type: Number,
    default: 0,
    validator: (value) => value >= 0
  },

  items: {
    type: Array,
    required: true,
    validator: (value) => value.length > 0
  }
})
</script>
```

### 2.3 TypeScript 类型验证

```vue
<script setup lang="ts">
type Size = 'small' | 'medium' | 'large'
type Status = 'active' | 'inactive'

interface Props {
  size?: Size
  status: Status
  count: number
  items: Array<{ id: number; label: string }>
}

const props = defineProps<Props>()
// TypeScript 编译时进行类型检查
</script>
```

## 三、注意事项与常见陷阱

- 验证函数在开发模式下运行，生产环境中不会执行
- 自定义验证函数返回 `false` 时会发出控制台警告
- TypeScript 类型声明提供编译时检查，运行时验证仍需要 `defineProps`
- 类型验证失败不会阻止组件渲染，只会发出警告
- 对于复杂类型验证，TypeScript 联合类型更合适
