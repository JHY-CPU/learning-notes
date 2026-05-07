# UI组件库Vuetify

## 一、概念说明

Vuetify 是基于 Material Design 规范的 Vue 3 UI 框架，提供 80+ 精美组件和强大的主题系统。Vuetify 3 完全重写为 TypeScript，支持 Tree Shaking 和动态主题切换，适合需要 Material Design 风格的项目。

## 二、具体用法

### 安装配置

```bash
npm create vuetify@latest my-vuetify-app
# 选择 preset: Default、Essentials、Blank
cd my-vuetify-app
npm install
npm run dev
```

```ts
// plugins/vuetify.ts
import 'vuetify/styles'
import { createVuetify } from 'vuetify'
import * as components from 'vuetify/components'
import * as directives from 'vuetify/directives'

export default createVuetify({
  components,
  directives,
  theme: {
    defaultTheme: 'light',
    themes: {
      light: {
        colors: {
          primary: '#1976D2',
          secondary: '#424242',
          accent: '#82B1FF'
        }
      },
      dark: {
        colors: {
          primary: '#2196F3'
        }
      }
    }
  }
})
```

### 基本组件使用

```vue
<script setup lang="ts">
const dialog = ref(false)
const selected = ref([])
const items = ['Vue', 'React', 'Angular', 'Svelte']

function toggleTheme() {
  // 动态切换主题
  console.log('切换主题')
}
</script>

<template>
  <v-app>
    <v-app-bar color="primary">
      <v-app-bar-title>我的应用</v-app-bar-title>
      <v-btn icon @click="toggleTheme">
        <v-icon>mdi-theme-light-dark</v-icon>
      </v-btn>
    </v-app-bar>

    <v-main>
      <v-container>
        <v-card class="mx-auto" max-width="400">
          <v-card-title>选择技术栈</v-card-title>
          <v-card-text>
            <v-chip-group v-model="selected" multiple>
              <v-chip v-for="item in items" :key="item" :value="item">
                {{ item }}
              </v-chip>
            </v-chip-group>
          </v-card-text>
          <v-card-actions>
            <v-btn color="primary" @click="dialog = true">详情</v-btn>
          </v-card-actions>
        </v-card>

        <v-dialog v-model="dialog" max-width="400">
          <v-card>
            <v-card-title>已选择</v-card-title>
            <v-card-text>{{ selected.join(', ') || '未选择' }}</v-card-text>
            <v-card-actions>
              <v-spacer />
              <v-btn @click="dialog = false">关闭</v-btn>
            </v-card-actions>
          </v-card>
        </v-dialog>
      </v-container>
    </v-main>
  </v-app>
</template>
```

### 响应式布局

```vue
<template>
  <v-container>
    <v-row>
      <!-- 移动端12列，平板6列，桌面4列 -->
      <v-col cols="12" md="6" lg="4" v-for="i in 6" :key="i">
        <v-card>
          <v-card-title>卡片 {{ i }}</v-card-title>
        </v-card>
      </v-col>
    </v-row>
  </v-container>
</template>
```

## 三、注意事项与常见陷阱

1. **Vuetify 3 包体积较大**：即使 Tree Shaking 后仍比其他 UI 库大
2. **Material Design 风格有学习曲线**：不熟悉 MD 规范的团队需要适应
3. **`v-app` 是必须的根组件**：所有 Vuetify 组件必须在 v-app 内
4. **图标需要安装 @mdi/font**：或使用 CDN 引入 Material Design Icons
5. **SASS 是依赖项**：Vuetify 需要 SASS 来编译样式
