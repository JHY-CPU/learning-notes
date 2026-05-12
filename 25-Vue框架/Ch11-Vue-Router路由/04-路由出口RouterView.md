# 路由出口RouterView

## 一、概念说明

`<RouterView>`是路由组件的渲染出口，告诉Vue Router在什么位置渲染匹配到的组件。它是一个函数式组件，可以出现在组件的任何位置。

```vue
<template>
  <header>
    <nav>
      <RouterLink to="/">首页</RouterLink>
      <RouterLink to="/about">关于</RouterLink>
    </nav>
  </header>

  <main>
    <RouterView />  <!-- 匹配的路由组件在这里渲染 -->
  </main>

  <footer>页脚</footer>
</template>
```

## 二、具体用法

### 命名视图

```vue
<template>
  <RouterView />               <!-- 默认视图 (name: default) -->
  <RouterView name="sidebar" /> <!-- 命名视图 -->
  <RouterView name="footer" />  <!-- 命名视图 -->
</template>
```

对应的路由配置：
```js
{
  path: '/dashboard',
  components: {
    default: DashboardMain,
    sidebar: DashboardSidebar,
    footer: DashboardFooter
  }
}
```

### 与transition配合

```vue
<template>
  <router-view v-slot="{ Component }">
    <transition name="fade" mode="out-in">
      <component :is="Component" />
    </transition>
  </router-view>
</template>
```

### 与keep-alive配合

```vue
<template>
  <router-view v-slot="{ Component }">
    <keep-alive include="Home,About">
      <component :is="Component" />
    </keep-alive>
  </router-view>
</template>
```

## 四、同时使用Transition + KeepAlive

```vue
<template>
  <router-view v-slot="{ Component, route }">
    <transition :name="route.meta.transition || 'fade'" mode="out-in">
      <keep-alive :include="cachedPages" :max="10">
        <component :is="Component" :key="route.path" />
      </keep-alive>
    </transition>
  </router-view>
</template>

<script setup>
import { ref } from 'vue'
const cachedPages = ref(['Home', 'Dashboard'])
</script>
```

## 五、RouterView的插槽属性

```vue
<template>
  <router-view v-slot="{ Component, route }">
    <!-- Component: 要渲染的组件 -->
    <!-- route: 当前路由对象 -->
    <div :key="route.fullPath">
      <component :is="Component" />
    </div>
  </router-view>
</template>
```

插槽解构属性：

| 属性 | 类型 | 说明 |
|------|------|------|
| `Component` | VNode | 要渲染的路由组件 |
| `route` | RouteLocationNormalized | 当前路由信息 |

## 三、注意事项与常见陷阱

1. `<RouterView>`没有根元素，不会渲染额外DOM
2. 嵌套路由中需要在子路由组件内再放一个`<RouterView>`
3. 路由组件被卸载时会触发`onUnmounted`，注意清理副作用
4. 使用`<keep-alive>`可缓存组件状态，避免重复创建
5. `v-slot`解构获取的`Component`是动态组件，需用`:is`绑定
6. KeepAlive包含的路由组件需要设置`name`选项
7. `key`设为`route.path`时路由参数变化也会重新创建组件
