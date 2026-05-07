# 路由元信息meta

## 一、概念说明

`meta`字段用于在路由配置中附加自定义信息，如页面标题、是否需要认证、角色权限等。这些信息在守卫和组件中都可以访问。

```js
const routes = [
  {
    path: '/',
    component: Home,
    meta: { title: '首页', requiresAuth: false }
  },
  {
    path: '/admin',
    component: Admin,
    meta: { title: '管理后台', requiresAuth: true, roles: ['admin'] }
  }
]
```

## 二、具体用法

### 在守卫中使用

```js
router.beforeEach((to) => {
  // 设置页面标题
  document.title = to.meta.title || '默认标题'

  // 检查认证
  if (to.meta.requiresAuth && !isAuthenticated) {
    return { path: '/login', query: { redirect: to.fullPath } }
  }

  // 检查角色
  if (to.meta.roles && !to.meta.roles.includes(getUserRole())) {
    return '/403'
  }
})
```

### 在组件中访问

```vue
<script setup>
import { useRoute } from 'vue-router'

const route = useRoute()

console.log(route.meta.title)
console.log(route.meta.requiresAuth)

// 混合所有父路由的meta
// 子路由meta会覆盖父路由同名属性
</script>
```

### 合并meta策略

```js
// 路由器配置中设置meta合并策略
const router = createRouter({
  history: createWebHistory(),
  routes,
  // 默认：子meta覆盖父meta
})

// 访问matched获取所有层级的meta
route.matched.forEach(record => {
  console.log(record.meta)
})
```

## 三、注意事项与常见陷阱

1. `meta`是路由配置的属性，不是组件的
2. 子路由的`meta`不会自动继承父路由的`meta`
3. `route.meta`是当前匹配路由的meta（已合并）
4. `route.matched`数组包含所有匹配的路由记录及其meta
5. TypeScript中可扩展`RouteMeta`接口获得类型支持
