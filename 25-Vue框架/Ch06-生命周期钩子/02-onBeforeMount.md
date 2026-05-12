# onBeforeMount

## 一、概念说明
`onBeforeMount` 在组件**挂载到 DOM 之前**调用。此时模板已编译完成（render 函数已生成），但 DOM 节点尚未创建并插入页面。

## 二、具体用法

### 2.1 基本用法
```vue
<script setup>
import { onBeforeMount, ref } from 'vue'

const status = ref('准备挂载')

onBeforeMount(() => {
  console.log('组件即将挂载')
  console.log('模板已编译，但 $el 还不存在')
  status.value = '即将挂载'
})
</script>
```

### 2.2 初始化非响应式数据
```vue
<script setup>
import { onBeforeMount } from 'vue'

// 在挂载前初始化第三方库配置
let chartInstance = null

onBeforeMount(() => {
  // 提前准备配置，但不操作 DOM
  chartInstance = { theme: 'dark', animate: true }
})
</script>
```

### 2.3 与 setup 的区别

```js
// setup: 最早执行，相当于 beforeCreate + created
// onBeforeMount: setup 之后、挂载之前

// 时间线：
// setup() -> onBeforeMount() -> DOM 创建 -> onMounted()
```

## 三、常见用例

| 场景 | 说明 |
|------|------|
| 准备初始数据 | 不依赖 DOM 的数据初始化 |
| 配置第三方库 | 提前设置参数，不操作 DOM |
| 条件逻辑 | 根据状态决定后续行为 |

## 四、注意事项与常见陷阱

- 此阶段**无法访问 DOM 元素**，不要尝试操作 `$el` 或 `ref` 绑定的 DOM
- 服务端渲染（SSR）中此钩子**会被调用**（注意：SSR 中 onMounted 不会执行）
- 大多数初始化逻辑应在 `onMounted` 中执行，而非 `onBeforeMount`
- 适合做不依赖 DOM 的准备工作
- 在 setup 中直接执行的代码与 onBeforeMount 效果相同

## 五、执行时机详解

```
组件创建流程：
  setup() 同步代码执行
    |
  template 编译为 render 函数
    |
  onBeforeMount() 调用  <-- 此时 render 函数可用，DOM 不可用
    |
  render 函数执行，创建虚拟 DOM
    |
  虚拟 DOM 挂载到真实 DOM
    |
  onMounted() 调用  <-- 此时 DOM 完全可用
```

## 六、实际使用建议

- 99% 的场景应该使用 `onMounted` 而非 `onBeforeMount`
- `onBeforeMount` 主要用于以下特殊场景：
  - 需要在渲染前修改响应式数据以影响首次渲染
  - SSR 中需要在服务端执行的初始化逻辑
  - 测试中需要在挂载前进行准备
