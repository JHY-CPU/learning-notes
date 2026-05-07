# Nuxt3组件自动导入

## 一、概念说明

Nuxt 3 默认自动导入 `components/`、`composables/`、`utils/` 目录下的模块，无需手动编写 import 语句。组件名基于文件路径自动生成，PascalCase 格式。这减少了样板代码，但也要求开发者遵循目录约定。

## 二、具体用法

### 组件自动导入

```
components/
├── AppHeader.vue        # → <AppHeader />
├── AppFooter.vue        # → <AppFooter />
├── user/
│   ├── ProfileCard.vue  # → <UserProfileCard />
│   └── Avatar.vue       # → <UserAvatar />
└── base/
    └── Button.vue       # → <BaseButton />
```

```vue
<!-- pages/index.vue -->
<!-- 无需 import，直接使用组件名 -->
<template>
  <div>
    <AppHeader />
    <UserProfileCard :user="currentUser" />
    <BaseButton @click="handleClick">点击我</BaseButton>
    <UserAvatar src="/avatar.png" size="large" />
    <AppFooter />
  </div>
</template>

<script setup>
// composables/ 下的函数也自动导入
const currentUser = ref({ name: '张三' })
const { data } = useMyCustomComposable()
// useMyCustomComposable 来自 composables/useMyCustomComposable.ts
</script>
```

### 懒加载组件

```vue
<!-- 组件名以 Lazy 前缀表示懒加载 -->
<template>
  <div>
    <LazyHeavyChart v-if="showChart" />
    <LazyComments v-if="showComments" />
    <!-- 组件只在条件为 true 时才加载 -->
  </div>
</template>

<script setup>
const showChart = ref(false)
const showComments = ref(false)
</script>
```

### 手动导入与禁用自动导入

```ts
// nuxt.config.ts
export default defineNuxtConfig({
  components: [
    {
      path: '~/components',
      pathPrefix: false,  // 禁用路径前缀，组件名就是文件名
      // 此时 user/ProfileCard.vue → <ProfileCard /> 而非 <UserProfileCard />
    }
  ]
})
```

```vue
<!-- 也可以手动导入某个组件 -->
<script setup>
import MySpecialButton from '~/components/special/MyButton.vue'
</script>

<template>
  <MySpecialButton>特殊按钮</MySpecialButton>
</template>
```

## 三、注意事项与常见陷阱

1. **组件名由路径决定**：`components/admin/UserTable.vue` 对应 `<AdminUserTable />`
2. **首字母必须大写**：组件文件名首字母小写会导致自动导入失败
3. **不要与 HTML 标签同名**：如 `components/Button.vue` 可能与原生 `<button>` 冲突
4. **自动导入有性能开销**：组件过多时可在配置中排除不需要的目录
5. **IDE 需要 Nuxt 插件支持**：VS Code 安装 Vue - Official 插件才能识别自动导入的类型
