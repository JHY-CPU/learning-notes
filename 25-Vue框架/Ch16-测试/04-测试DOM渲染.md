# 测试 DOM 渲染

## 一、概念说明

测试组件是否正确渲染了预期的 DOM 结构是组件测试的基础。Vue Test Utils 提供了多种方式查找和断言 DOM 元素。

```vue
<!-- UserCard.vue -->
<script setup>
defineProps({
  user: { type: Object, required: true },
  isActive: { type: Boolean, default: false }
})
</script>

<template>
  <div class="user-card" :class="{ active: isActive }">
    <h3 data-testid="username">{{ user.name }}</h3>
    <p class="email">{{ user.email }}</p>
    <span v-if="user.isAdmin" class="badge">管理员</span>
  </div>
</template>
```

```js
// UserCard.test.js
import { mount } from '@vue/test-utils'
import UserCard from '../UserCard.vue'

describe('UserCard', () => {
  const user = { name: '张三', email: 'zhang@example.com', isAdmin: true }

  it('渲染用户名', () => {
    const wrapper = mount(UserCard, { props: { user } })
    expect(wrapper.find('[data-testid="username"]').text()).toBe('张三')
  })

  it('渲染邮箱', () => {
    const wrapper = mount(UserCard, { props: { user } })
    expect(wrapper.find('.email').text()).toBe('zhang@example.com')
  })

  it('管理员显示徽章', () => {
    const wrapper = mount(UserCard, { props: { user } })
    expect(wrapper.find('.badge').exists()).toBe(true)
  })

  it('非管理员不显示徽章', () => {
    const wrapper = mount(UserCard, {
      props: { user: { ...user, isAdmin: false } }
    })
    expect(wrapper.find('.badge').exists()).toBe(false)
  })
})
```

## 二、具体用法

### 2.1 查找元素

```js
wrapper.find('.class')           // CSS 选择器
wrapper.find('#id')              // ID 选择器
wrapper.find('[data-testid="x"]') // data 属性
wrapper.findComponent(ChildComp) // 查找子组件
wrapper.findAll('.item').length  // 查找所有匹配
```

### 2.2 断言方法

```js
expect(wrapper.find('.box').exists()).toBe(true)       // 存在
expect(wrapper.find('.box').isVisible()).toBe(true)     // 可见
expect(wrapper.find('input').element.value).toBe('123') // input 值
expect(wrapper.classes()).toContain('active')           // CSS 类
expect(wrapper.attributes('id')).toBe('main')           // 属性
```

## 三、注意事项与常见陷阱

- 优先使用 `data-testid` 选择元素，避免耦合 CSS 类名
- `wrapper.element` 是原生 DOM 元素，`wrapper.vm` 是 Vue 组件实例
- 异步更新后需要 `await wrapper.vm.$nextTick()` 或 `await wrapper.setProps()`
