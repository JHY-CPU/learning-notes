# 组合式函数 vs React Hooks

## 一、概念说明

Vue组合式函数和React Hooks在设计目标上相似（逻辑复用），但实现原理不同。组合式函数只在setup中调用一次，而React Hooks在每次渲染时都调用。

```vue
<!-- Vue 组合式函数 -->
<script setup>
import { useMouse } from './composables/useMouse'
const { x, y } = useMouse() // 只调用一次，自动响应式
</script>
```

```jsx
// React Hooks
function Component() {
  const { x, y } = useMouse() // 每次渲染都调用
  return <div>{x}, {y}</div>
}
```

## 二、具体用法

### 实现对比

```js
// Vue 组合式函数 - 无顺序限制
export function useMouse() {
  const x = ref(0)
  const y = ref(0)
  // 可以有条件调用
  if (someCondition) {
    const z = ref(0)
  }
  return { x, y }
}
```

```jsx
// React Hooks - 顺序敏感
function useMouse() {
  const [x, setX] = useState(0)
  const [y, setY] = useState(0)
  // 不能有条件调用！必须保持调用顺序一致
  return { x, y }
}
```

### 核心差异

| 特性 | Vue组合式函数 | React Hooks |
|------|--------------|-------------|
| 调用次数 | setup时调用一次 | 每次渲染都调用 |
| 条件调用 | 可以 | 不可以 |
| 调用顺序 | 无限制 | 必须一致 |
| 响应式机制 | Proxy自动追踪 | 手动依赖数组 |
| 清理函数 | onUnmounted | useEffect返回函数 |
| 性能开销 | 低（Proxy懒追踪） | 较高（每次重新渲染） |

### 依赖追踪对比

```js
// Vue - 自动追踪依赖
const doubled = computed(() => count.value * 2)

// React - 手动声明依赖
const doubled = useMemo(() => count * 2, [count])
```

## 三、注意事项与常见陷阱

1. Vue组合式函数不需要遵循React Hooks的两条规则（顶层调用、仅在函数组件中调用）
2. 组合式函数中的`ref`是持久引用，不会在渲染间重建
3. React的`useEffect`需要依赖数组，Vue的`watch`默认懒执行
4. 组合式函数可在普通JS文件中使用，不依赖组件上下文
5. Vue的响应式系统是基于Proxy的，更细粒度的依赖追踪
