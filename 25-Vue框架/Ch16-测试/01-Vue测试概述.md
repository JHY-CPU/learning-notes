# Vue 测试概述

## 一、概念说明

测试是保证代码质量的重要手段。Vue 项目的测试遵循**测试金字塔**原则：底层是大量快速的单元测试，中层是集成测试，顶层是少量 E2E 测试。

```
        /  E2E 测试  \        ← 少量，模拟真实用户操作
       /  集成测试     \       ← 中等，测试组件协作
      /   单元测试      \      ← 大量，测试独立函数/组件
```

| 测试类型 | 速度 | 覆盖范围 | 工具 |
|---------|------|---------|------|
| 单元测试 | 快 | 函数、组件 | Vitest + Vue Test Utils |
| 集成测试 | 中 | 多组件协作 | Vitest + Testing Library |
| E2E 测试 | 慢 | 完整用户流程 | Cypress / Playwright |

```js
// 一个简单的单元测试示例
import { describe, it, expect } from 'vitest'
import { mount } from '@vue/test-utils'
import Counter from '../Counter.vue'

describe('Counter', () => {
  it('点击按钮后计数增加', async () => {
    const wrapper = mount(Counter)
    await wrapper.find('button').trigger('click')
    expect(wrapper.text()).toContain('1')
  })
})
```

## 二、具体用法

### 2.1 测试金字塔原则

- 单元测试：70%，测试独立的函数和组件
- 集成测试：20%，测试组件之间的交互
- E2E 测试：10%，测试关键业务流程

### 2.2 Vue 生态测试工具

- **Vitest**：Vite 原生测试框架，速度快
- **Vue Test Utils**：Vue 官方组件测试工具
- **Cypress / Playwright**：E2E 测试框架

## 三、注意事项与常见陷阱

- 不要追求 100% 覆盖率，关注关键业务逻辑
- 测试行为而非实现，避免测试内部细节
- 每个测试应该独立，不依赖其他测试的结果

## 四、测试配置

```js
// vitest.config.ts
import { defineConfig } from 'vitest/config'
import vue from '@vitejs/plugin-vue'

export default defineConfig({
  plugins: [vue()],
  test: {
    environment: 'jsdom',            // 模拟浏览器环境
    globals: true,                   // 全局使用 describe/it/expect
    include: ['**/*.{test,spec}.{js,ts,jsx,tsx}'],
    coverage: {
      reporter: ['text', 'html'],    // 覆盖率报告
      exclude: ['node_modules/']
    }
  }
})
```

## 五、测试优先级

```
优先测试：
  1. 业务逻辑（纯函数、计算属性）
  2. 用户交互（表单提交、按钮点击）
  3. 条件渲染（v-if 分支）
  4. 数据展示（props 传递）

次要测试：
  5. 样式变化
  6. 动画效果
  7. 第三方库集成
```

## 六、测试命名规范

```js
// ✅ 好的命名
describe('UserForm', () => {
  it('当邮箱为空时显示错误提示', () => {})
  it('提交成功后清空表单', () => {})
  it('密码少于8位时禁用提交按钮', () => {})
})

// ❌ 不好的命名
describe('UserForm', () => {
  it('测试验证', () => {})
  it('测试提交', () => {})
})
```
