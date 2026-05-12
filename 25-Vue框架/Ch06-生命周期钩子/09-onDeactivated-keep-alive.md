# onDeactivated keep-alive

## 一、概念说明
`onDeactivated` 在被 `<keep-alive>` 缓存的组件**失活时**调用。当组件被切换走但并未销毁（仍在缓存中）时，此钩子会被触发。

## 二、具体用法

### 2.1 基本用法
```vue
<script setup>
import { onDeactivated, ref } from 'vue'

const status = ref('活跃')

onDeactivated(() => {
  console.log('组件失活（进入缓存）')
  status.value = '已缓存'
})
</script>
```

### 2.2 暂停耗时操作
```vue
<script setup>
import { onActivated, onDeactivated } from 'vue'

let animationId = null

function animate() {
  animationId = requestAnimationFrame(animate)
  // 动画逻辑...
}

onActivated(() => { animate() })
onDeactivated(() => { cancelAnimationFrame(animationId) })
</script>
```

### 2.3 保存页面状态
```vue
<script setup>
import { ref, onDeactivated } from 'vue'
import { usePageStore } from '@/stores/page'

const store = usePageStore()
const formData = ref({ name: '', email: '' })

onDeactivated(() => {
  store.saveForm(formData.value)
})
</script>
```

## 三、注意事项与常见陷阱
- 失活不等于卸载，组件实例仍在内存中
- 不要在 `onDeactivated` 中执行销毁操作
- 与 `onUnmounted` 的区别：`onDeactivated` 组件可被恢复
- 此钩子在服务端渲染中不会被调用

## 四、执行时机

```
组件生命周期（keep-alive）：
  首次创建：setup -> onBeforeMount -> onMounted -> onActivated
  切走：onDeactivated
  切回：onActivated
  再切走：onDeactivated
  最终卸载（被 exclude 或 max 淘汰）：onBeforeUnmount -> onUnmounted
```

## 五、常见使用场景

- 暂停轮询和定时器
- 暂停 requestAnimationFrame 动画
- 保存表单数据到 store
- 取消未完成的网络请求

## 六、onDeactivated vs onBeforeUnmount

| 场景 | onDeactivated | onBeforeUnmount |
| --- | --- | --- |
| 组件还在内存中？ | 是 | 否（即将移除） |
| 会被恢复？ | 是 | 否 |
| 清理方式 | 暂停 | 彻底清理 |
| 适用 | keep-alive | 所有组件 |

## 七、完整暂停/恢复模式

```vue
<script setup>
import { onActivated, onDeactivated, onUnmounted } from 'vue'

let pollTimer = null
let ws = null
let animId = null

// 激活时启动所有资源
onActivated(() => {
  pollTimer = setInterval(fetchData, 5000)
  ws = new WebSocket('wss://example.com')
  function tick() { animId = requestAnimationFrame(tick) }
  tick()
})

// 失活时暂停所有资源
onDeactivated(() => {
  clearInterval(pollTimer)   // 暂停轮询
  ws?.close()                // 关闭连接
  cancelAnimationFrame(animId) // 暂停动画
})

// 最终卸载时确认清理
onUnmounted(() => {
  clearInterval(pollTimer)
  ws?.close()
  cancelAnimationFrame(animId)
})
</script>
```

## 八、Vue Router 中的 keep-alive 失活

```vue
<!-- 使用 route.meta 控制缓存 -->
<router-view v-slot="{ Component, route }">
  <keep-alive :include="cachedRoutes">
    <component :is="Component" :key="route.name" />
  </keep-alive>
</router-view>

<!-- 组件离开路由时会触发 onDeactivated -->
<!-- 返回时触发 onActivated -->
```
