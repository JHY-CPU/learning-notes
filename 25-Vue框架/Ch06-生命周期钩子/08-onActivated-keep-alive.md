# onActivated keep-alive

## 一、概念说明
`onActivated` 在被 `<keep-alive>` 缓存的组件**激活时**调用。当组件从缓存中恢复并重新显示时，此钩子会被触发，适合恢复组件状态或重新启动操作。

## 二、具体用法

### 2.1 基本用法
```vue
<!-- 父组件 -->
<template>
  <keep-alive>
    <ComponentA v-if="showA" />
    <ComponentB v-else />
  </keep-alive>
  <button @click="showA = !showA">切换</button>
</template>
```

```vue
<!-- ComponentA.vue -->
<script setup>
import { onActivated, onMounted, ref } from 'vue'

const status = ref('未激活')

onMounted(() => { console.log('首次挂载') })
onActivated(() => {
  console.log('组件被激活（从缓存恢复）')
  status.value = '已激活'
})
</script>
```

### 2.2 恢复定时器/轮询
```vue
<script setup>
import { onActivated, onDeactivated } from 'vue'

let pollTimer = null

onActivated(() => {
  pollTimer = setInterval(fetchData, 5000)
})

onDeactivated(() => {
  clearInterval(pollTimer)
})
</script>
```

### 2.3 恢复滚动位置
```vue
<script setup>
import { ref, onActivated, onDeactivated } from 'vue'

const scrollPos = ref(0)
const containerRef = ref(null)

onDeactivated(() => {
  scrollPos.value = containerRef.value?.scrollTop || 0
})

onActivated(() => {
  containerRef.value?.scrollTo(0, scrollPos.value)
})
</script>
```

## 三、注意事项与常见陷阱
- 只有被 `<keep-alive>` 包裹的组件才会触发此钩子
- 首次挂载时 `onMounted` 和 `onActivated` 都会触发
- 之后每次从缓存恢复只触发 `onActivated`
- 此钩子在服务端渲染中不会被调用

## 四、执行时机对照

```
首次挂载：onBeforeMount -> onMounted -> onActivated
切走：onDeactivated
切回：onActivated（不再触发 onMounted）
最终销毁：onBeforeUnmount -> onUnmounted
```

## 五、常见使用场景

- 恢复列表滚动位置
- 重新启动数据轮询
- 刷新过期数据
- 恢复动画

## 六、keep-alive 配置

```vue
<template>
  <!-- include/exclude 按组件名匹配 -->
  <keep-alive include="ComponentA,ComponentB">
    <component :is="currentTab" />
  </keep-alive>

  <!-- 最多缓存 5 个组件实例 -->
  <keep-alive :max="5">
    <router-view />
  </keep-alive>

  <!-- 使用正则 -->
  <keep-alive :include="/Tab[A-Z]/">
    <router-view />
  </keep-alive>
</template>
```

## 七、Vue Router 集成

```vue
<!-- 路由级别 keep-alive -->
<template>
  <router-view v-slot="{ Component }">
    <keep-alive :include="cachedViews">
      <component :is="Component" />
    </keep-alive>
  </router-view>
</template>

<script setup>
import { ref } from 'vue'

const cachedViews = ref(['Home', 'UserList'])

// 动态管理缓存
function addCache(name) {
  if (!cachedViews.value.includes(name)) cachedViews.value.push(name)
}
function removeCache(name) {
  cachedViews.value = cachedViews.value.filter(v => v !== name)
}
</script>
```

## 八、调试技巧

- 在 `onActivated` 和 `onDeactivated` 中打印日志确认缓存是否生效
- 使用 Vue DevTools 查看组件是否在 keep-alive 缓存中
- 如果组件没有被缓存，检查 `name` 选项是否与 include 匹配
