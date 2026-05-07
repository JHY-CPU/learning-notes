# 国际化vue-i18n

## 一、概念说明

vue-i18n 是 Vue 官方的国际化（i18n）插件，支持多语言切换、复数、日期/数字格式化。Vue I18n 9.x 专为 Vue 3 设计，支持 Composition API 和 TypeScript 类型安全。

## 二、具体用法

### 安装配置

```bash
npm install vue-i18n@9
```

```ts
// i18n/index.ts
import { createI18n } from 'vue-i18n'

const messages = {
  zh: {
    greeting: '你好，{name}！',
    menu: {
      home: '首页',
      about: '关于'
    },
    items: '没有项目 | 1 个项目 | {count} 个项目'
  },
  en: {
    greeting: 'Hello, {name}!',
    menu: {
      home: 'Home',
      about: 'About'
    },
    items: 'No items | 1 item | {count} items'
  }
}

export const i18n = createI18n({
  legacy: false,        // 使用 Composition API 模式
  locale: 'zh',         // 默认语言
  fallbackLocale: 'en', // 回退语言
  messages
})
```

### 组件中使用

```vue
<script setup lang="ts">
import { useI18n } from 'vue-i18n'

const { t, locale, availableLocales } = useI18n()

// 基本翻译
console.log(t('menu.home'))
// 输出：首页

// 带参数
console.log(t('greeting', { name: '张三' }))
// 输出：你好，张三！

// 复数
console.log(t('items', 0))
// 输出：没有项目
console.log(t('items', 5))
// 输出：5 个项目

// 切换语言
function switchLocale(lang: string) {
  locale.value = lang
  // 切换后所有 t() 输出变为对应语言
}
</script>

<template>
  <div>
    <nav>
      <button v-for="lang in availableLocales" :key="lang"
        @click="switchLocale(lang)"
        :class="{ active: locale === lang }">
        {{ lang === 'zh' ? '中文' : 'English' }}
      </button>
    </nav>

    <h1>{{ t('menu.home') }}</h1>
    <p>{{ t('greeting', { name: 'Vue 开发者' }) }}</p>
    <p>{{ t('items', { count: 10 }) }}</p>
  </div>
</template>
```

### 按需加载语言包

```ts
// i18n/index.ts - 懒加载
import { createI18n } from 'vue-i18n'

const i18n = createI18n({
  legacy: false,
  locale: navigator.language.split('-')[0],
  messages: {}
})

// 动态加载语言包
async function loadLocaleMessages(locale: string) {
  const messages = await import(`./locales/${locale}.json`)
  i18n.global.setLocaleMessage(locale, messages.default)
  return nextTick()
}

export { i18n, loadLocaleMessages }
```

```vue
<script setup lang="ts">
import { loadLocaleMessages } from '@/i18n'

async function switchLocale(lang: string) {
  await loadLocaleMessages(lang)
  locale.value = lang
}
</script>
```

### 日期和数字格式化

```ts
// 自定义格式化
const i18n = createI18n({
  // ...
  datetimeFormats: {
    zh: {
      short: { year: 'numeric', month: 'short', day: 'numeric' }
    }
  },
  numberFormats: {
    zh: {
      currency: { style: 'currency', currency: 'CNY' }
    }
  }
})
```

```vue
<script setup lang="ts">
const { d, n } = useI18n()
// d(new Date()) → 2024年1月15日
// n(99.9, 'currency') → ¥99.90
</script>
```

## 三、注意事项与常见陷阱

1. **legacy: false 是必须的**：不设置则无法在 setup 中使用 useI18n
2. **key 命名建议用点号**：`menu.home` 而非 `menuHome`，便于组织
3. **复数规则因语言而异**：中文没有复数变化，需要自定义复数函数
4. **动态切换语言要 await nextTick**：确保 DOM 更新完成
5. **SEO 场景需要 SSR 支持**：Nuxt 中使用 @nuxtjs/i18n 模块
