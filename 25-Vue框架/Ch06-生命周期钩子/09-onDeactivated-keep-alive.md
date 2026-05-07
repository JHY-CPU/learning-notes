# onDeactivated keep-alive

## 一、概念说明
`onDeactivated` 在被 `<keep-alive>` 缓存的组件**失活时**调用。当组件被切换走但并未销毁（仍在缓存中）时，此钩子会被触发。

## 二、具体用法

### 2.1 基本用法
```vue
<!-- ComponentA.vue -->
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

onActivated(() => {
  animate()  // 激活时启动动画
})

onDeactivated(() => {
  cancelAnimationFrame(animationId)  // 失活时暂停动画
})
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
  // 切走时保存表单数据到 store
  store.saveForm(formData.value)
})
</script>
```

## 三、注意事项与常见陷阱
- 失活不等于卸载，组件实例仍在内存中
- 不要在 `onDeactivated` 中执行销毁操作（如 `clearInterval` 只暂停不销毁）
- 与 `onUnmounted` 的区别：`onDeactivated` 组件可被恢复
- 此钩子在服务端渲染中不会被调用
