# 组件测试 Vue Test Utils

## 一、概念说明

Vue Test Utils 是 Vue 官方的组件测试工具库，提供 `mount` 和 `shallowMount` 方法将组件渲染到虚拟 DOM 中，方便测试组件的输出和行为。

```vue
<!-- Counter.vue -->
<script setup>
import { ref } from 'vue'
const count = ref(0)
defineEmits(['change'])
function increment() {
  count.value++
}
</script>

<template>
  <button @click="increment">计数: {{ count }}</button>
</template>
```

```js
// Counter.test.js
import { describe, it, expect } from 'vitest'
import { mount, shallowMount } from '@vue/test-utils'
import Counter from '../Counter.vue'

describe('Counter', () => {
  it('正确渲染初始值', () => {
    const wrapper = mount(Counter)
    expect(wrapper.text()).toContain('计数: 0')
  })

  it('点击后计数增加', async () => {
    const wrapper = mount(Counter)
    await wrapper.find('button').trigger('click')
    expect(wrapper.text()).toContain('计数: 1')
  })
})
```

## 二、具体用法

### 2.1 mount vs shallowMount

```js
// mount：完整渲染所有子组件
const full = mount(Parent)

// shallowMount：子组件替换为 stub，只测试当前组件
const shallow = shallowMount(Parent)
```

### 2.2 常用 API

```js
wrapper.text()              // 获取文本内容
wrapper.html()              // 获取 HTML
wrapper.find('.class')      // 查找元素
wrapper.findAll('button')   // 查找所有匹配元素
wrapper.trigger('click')    // 触发事件
wrapper.vm                  // 访问组件实例
wrapper.props()             // 获取 props
wrapper.emitted()           // 获取已触发的事件
```

### 2.3 挂载选项

```js
const wrapper = mount(Counter, {
  props: { initialCount: 10 },
  slots: { default: '<span>插槽内容</span>' },
  global: {
    plugins: [pinia, router],
    stubs: ['RouterLink']
  }
})
```

## 三、注意事项与常见陷阱

- 触发事件后必须使用 `await` 等待 DOM 更新
- `shallowMount` 适合隔离测试，`mount` 适合集成测试
- 组件中使用了 Composition API 需要 `jsdom` 环境
