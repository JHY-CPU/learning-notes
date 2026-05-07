# KeepAlive 缓存

## 一、概念说明

`<KeepAlive>` 是 Vue 内置组件，用于缓存不活动的组件实例，避免重复创建和销毁。常用于标签页切换、路由缓存等场景。

```vue
<script setup>
import { ref, shallowRef } from 'vue'
import TabA from './TabA.vue'
import TabB from './TabB.vue'
import TabC from './TabC.vue'

const currentTab = shallowRef(TabA)
const tabs = { TabA, TabB, TabC }
</script>

<template>
  <button v-for="(_, name) in tabs" :key="name" @click="currentTab = tabs[name]">
    {{ name }}
  </button>

  <KeepAlive>
    <component :is="currentTab" />
  </KeepAlive>
</template>
```

## 二、具体用法

### 2.1 缓存控制

```vue
<!-- 只缓存 TabA 和 TabB -->
<KeepAlive include="TabA,TabB">
  <component :is="currentTab" />
</KeepAlive>

<!-- 缓存匹配正则的组件 -->
<KeepAlive :include="/^Cached/">
  <router-view />
</KeepAlive>

<!-- 排除某些组件 -->
<KeepAlive exclude="NoCache">
  <router-view />
</KeepAlive>
```

### 2.2 最大缓存数

```vue
<!-- 最多缓存 5 个组件实例，超出时销毁最久未访问的 -->
<KeepAlive :max="5">
  <router-view />
</KeepAlive>
```

### 2.3 生命周期钩子

```vue
<script setup>
import { onActivated, onDeactivated } from 'vue'

// 组件被缓存后重新激活
onActivated(() => {
  console.log('组件激活，刷新数据')
  fetchData()
})

// 组件被缓存时（离开）
onDeactivated(() => {
  console.log('组件被缓存')
})
</script>
```

### 2.4 路由缓存

```vue
<RouterView v-slot="{ Component, route }">
  <KeepAlive :include="cachedViews">
    <component :is="Component" :key="route.path" />
  </KeepAlive>
</RouterView>
```

## 三、注意事项与常见陷阱

- `include` 匹配的是组件的 `name` 选项，`<script setup>` 需手动设置 `defineOptions({ name: 'Xxx' })`
- 缓存的组件不会触发 `mounted`/`unmounted`，使用 `onActivated`/`onDeactivated`
- 缓存会占用内存，设置合理的 `max` 限制
- 缓存中组件的数据可能过时，需要在 `onActivated` 中刷新
