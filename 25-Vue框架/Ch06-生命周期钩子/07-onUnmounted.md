# onUnmounted

## 一、概念说明
`onUnmounted` 在组件**完全卸载之后**调用。此时组件的 DOM 已被移除，所有响应式副作用（computed、watch、effect）已停止。这是组件生命周期的最后一个阶段。

## 二、具体用法

### 2.1 基本用法
```vue
<script setup>
import { onUnmounted } from 'vue'

onUnmounted(() => {
  console.log('组件已完全卸载')
  console.log('DOM 已移除，响应式已停止')
})
</script>
```

### 2.2 最终清理
```vue
<script setup>
import { onMounted, onUnmounted } from 'vue'

let connection = null

onMounted(() => {
  connection = new WebSocket('ws://example.com')
})

onUnmounted(() => {
  if (connection) {
    connection.close()
    connection = null
  }
})
</script>
```

### 2.3 清理全局状态
```vue
<script setup>
import { onUnmounted } from 'vue'
import { useAppStore } from '@/stores/app'

const store = useAppStore()

onUnmounted(() => {
  store.clearComponentData()
})
</script>
```

## 三、注意事项与常见陷阱
- `onUnmounted` 中**不能再访问响应式数据的更新**
- 与 `onBeforeUnmount` 的区别：前者组件仍在，后者已完全销毁
- 此钩子在服务端渲染中不会被调用
- 推荐在 `onBeforeUnmount` 中做清理，`onUnmounted` 做日志记录

## 四、与 onBeforeUnmount 的对比

| 特性 | onBeforeUnmount | onUnmounted |
| --- | --- | --- |
| 时机 | DOM 移除前 | DOM 移除后 |
| 组件可用性 | 完全可用 | DOM 已移除 |
| 响应式数据 | 可访问 | 不可更新 |
| 推荐用途 | 清理副作用 | 日志记录 |
| 第三方库清理 | 最佳时机 | 不推荐 |

## 五、实际使用建议

- 清理工作优先在 `onBeforeUnmount` 完成
- `onUnmounted` 适合做最后的日志记录或调试输出
- 不要在 `onUnmounted` 中依赖 DOM 或响应式数据

## 六、测试中的使用

```js
// 在单元测试中验证清理逻辑
import { mount } from '@vue/test-utils'
import { vi } from 'vitest'

test('组件卸载时清理定时器', () => {
  const clearIntervalSpy = vi.spyOn(global, 'clearInterval')
  const wrapper = mount(MyComponent)
  wrapper.unmount()
  expect(clearIntervalSpy).toHaveBeenCalled()
})
```

## 七、执行时机

```
组件卸载流程：
  onBeforeUnmount()  // 组件仍可用
    |
  移除 DOM 节点
    |
  停止响应式副作用（computed, watch, effect）
    |
  onUnmounted()  // 最后一个钩子
    |
  组件实例被 GC 回收
```
