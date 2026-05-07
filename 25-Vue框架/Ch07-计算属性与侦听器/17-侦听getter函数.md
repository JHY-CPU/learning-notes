# 侦听 getter 函数

## 一、概念说明
`watch` 的第一个参数可以是一个**getter 函数**，而非直接传 ref。这在需要侦听 reactive 对象的某个属性、computed 值或复杂表达式时非常有用。

## 二、具体用法

### 2.1 侦听 reactive 属性
```vue
<script setup>
import { reactive, watch } from 'vue'

const state = reactive({ name: '张三', age: 25 })

// 使用 getter 侦听特定属性
watch(
  () => state.name,
  (newVal, oldVal) => {
    console.log(`名字从 ${oldVal} 变为 ${newVal}`)
  }
)
</script>
```

### 2.2 侦听 computed
```vue
<script setup>
import { ref, computed, watch } from 'vue'

const firstName = ref('张')
const lastName = ref('三')
const fullName = computed(() => `${firstName.value} ${lastName.value}`)

watch(fullName, (newVal) => {
  console.log('全名变为:', newVal)
})

// 等价于 getter 写法
watch(() => `${firstName.value} ${lastName.value}`, (newVal) => {
  console.log('全名变为:', newVal)
})
</script>
```

### 2.3 侦听复杂表达式
```vue
<script setup>
import { ref, watch } from 'vue'

const min = ref(0)
const max = ref(100)

watch(
  () => max.value - min.value,
  (range) => {
    console.log('范围变为:', range)
  }
)
</script>
```

### 2.4 侦听嵌套属性
```vue
<script setup>
import { ref, watch } from 'vue'

const user = ref({
  profile: { address: { city: '北京' } }
})

watch(
  () => user.value.profile.address.city,
  (newCity) => {
    console.log('城市变为:', newCity)
  }
)
</script>
```

## 三、注意事项与常见陷阱
- getter 函数中访问的属性必须是响应式的
- 使用 getter 可以获取正确的 oldValue（直接侦听 reactive 对象做不到）
- getter 返回的值作为 watch 的数据源，类型应保持一致
- 推荐用 getter 侦听 reactive 的属性，而非直接侦听整个 reactive 对象
