# watch 选项 deep

## 一、概念说明
`deep` 选项控制 watch 是否**深度侦听**对象的嵌套属性变化。对于 ref 包装的对象，默认只侦听引用变化；加上 `deep: true` 后可以侦听内部属性的变化。

## 二、具体用法

### 2.1 ref 对象需要 deep
```vue
<script setup>
import { ref, watch } from 'vue'

const user = ref({ name: '张三', address: { city: '北京' } })

// ❌ 不加 deep: 只侦听引用变化
watch(user, (val) => {
  console.log('变化了')  // 修改 user.value.name 不会触发
})

// ✅ 加 deep: 侦听所有嵌套属性
watch(user, (val) => {
  console.log('深度变化:', val)
}, { deep: true })
</script>
```

### 2.2 侦听嵌套属性的精确控制
```vue
<script setup>
import { ref, watch } from 'vue'

const config = ref({
  theme: { color: 'blue', size: 14 },
  layout: { sidebar: true }
})

// 只侦听特定嵌套属性（更高效）
watch(
  () => config.value.theme.color,
  (newColor) => {
    document.body.style.color = newColor
  }
)
</script>
```

### 2.3 深度侦听的性能影响
```vue
<script setup>
import { reactive, watch } from 'vue'

// 大对象深度侦听有性能开销
const largeData = reactive({ /* 大量嵌套数据 */ })

// ❌ 深度侦听整个大对象
watch(largeData, cb, { deep: true })

// ✅ 只侦听需要的属性
watch(() => largeData.specificField, cb)
</script>
```

## 三、注意事项与常见陷阱
- `reactive` 对象默认就是深度侦听的，不需要加 `deep: true`
- `ref` 包装的对象需要加 `deep: true` 才能侦听嵌套属性
- 深度侦听有性能开销，大对象慎用
- 优先使用 getter 函数精确侦听特定属性，而非深度侦听整个对象
