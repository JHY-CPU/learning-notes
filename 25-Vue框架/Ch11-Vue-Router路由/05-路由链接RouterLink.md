# 路由链接RouterLink

## 一、概念说明

`<RouterLink>`是用于导航的组件，渲染为`<a>`标签。它提供激活状态样式、预加载等功能，替代手动使用`<a>`标签。

```vue
<template>
  <nav>
    <RouterLink to="/">首页</RouterLink>
    <RouterLink to="/about" active-class="current">关于</RouterLink>
    <RouterLink :to="{ name: 'User', params: { id: 123 } }">用户</RouterLink>
  </nav>
</template>
```

## 二、具体用法

### 属性说明

```vue
<!-- 字符串路径 -->
<RouterLink to="/about">关于</RouterLink>

<!-- 对象形式 -->
<RouterLink :to="{ path: '/about' }">关于</RouterLink>
<RouterLink :to="{ name: 'User', params: { id: 1 } }">用户</RouterLink>
<RouterLink :to="{ path: '/search', query: { q: 'vue' } }">搜索</RouterLink>

<!-- 自定义激活类名 -->
<RouterLink to="/about" active-class="active">关于</RouterLink>
<RouterLink to="/about" exact-active-class="exact-active">关于</RouterLink>

<!-- 替换历史记录 -->
<RouterLink to="/login" replace>登录</RouterLink>

<!-- 外部链接 -->
<a href="https://vuejs.org" target="_blank">Vue官网</a>
```

### 自定义渲染

```vue
<RouterLink to="/about" custom v-slot="{ href, navigate, isActive }">
  <li :class="{ active: isActive }" @click="navigate">
    <a :href="href">关于</a>
  </li>
</RouterLink>
```

### CSS类名

| 类名 | 说明 |
|------|------|
| `router-link-active` | 路径包含目标时激活 |
| `router-link-exact-active` | 路径完全匹配时激活 |

## 三、注意事项与常见陷阱

1. `RouterLink`会自动阻止默认行为，使用客户端导航
2. 外部链接使用普通`<a>`标签，不要用`RouterLink`
3. `active-class`是路径部分匹配时的类名
4. `exact-active-class`是路径完全匹配时的类名
5. 可以通过全局配置设置默认的`active-class`
