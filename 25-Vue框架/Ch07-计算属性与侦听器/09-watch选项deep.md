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

## 四、deep 的性能影响与优化

### 4.1 性能测试对比
```js
import { ref, watch } from 'vue'

// 小对象：deep 影响不大
const smallObj = ref({ name: 'test', value: 42 })
watch(smallObj, cb, { deep: true })  // OK

// 大对象：deep 有明显开销
const largeObj = ref({
  items: Array.from({ length: 10000 }, (_, i) => ({
    id: i, data: { nested: { value: i } }
  }))
})
// ❌ 深度侦听 10000 个嵌套对象
watch(largeObj, cb, { deep: true })

// ✅ 只侦听需要的部分
watch(() => largeObj.value.items.length, cb)
```

### 4.2 使用 shallowRef 替代
```vue
<script setup>
import { shallowRef, triggerRef } from 'vue'

// shallowRef 不会深度追踪
const state = shallowRef({ nested: { value: 1 } })

// 修改嵌套属性不会触发更新
state.value.nested.value = 2  // 不触发

// 需要手动触发
state.value = { nested: { value: 2 } }  // 触发
// 或
triggerRef(state)  // 强制触发
</script>
```

### 4.3 自定义深比较
```vue
<script setup>
import { ref, watch } from 'vue'

const config = ref({ theme: 'dark', fontSize: 14 })

// 使用自定义比较函数代替 deep
watch(config, (newVal, oldVal) => {
  if (newVal.theme !== oldVal?.theme) {
    applyTheme(newVal.theme)
  }
  if (newVal.fontSize !== oldVal?.fontSize) {
    applyFontSize(newVal.fontSize)
  }
}, {
  deep: false  // 不用 deep，手动比较
})
</script>
```
