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
