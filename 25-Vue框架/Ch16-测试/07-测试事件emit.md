# 测试事件 emit

## 一、概念说明

Vue 组件通过 `emit` 向父组件发送事件。测试 emit 可以验证组件是否在正确时机触发了正确参数的事件。

```vue
<!-- TodoItem.vue -->
<script setup>
const props = defineProps({ todo: Object })
const emit = defineEmits(['toggle', 'delete'])

function onToggle() {
  emit('toggle', props.todo.id)
}

function onDelete() {
  emit('delete', props.todo.id)
}
</script>

<template>
  <div :class="{ done: todo.completed }">
    <input type="checkbox" :checked="todo.completed" @change="onToggle" />
    <span>{{ todo.text }}</span>
    <button @click="onDelete">删除</button>
  </div>
</template>
```

```js
import { mount } from '@vue/test-utils'
import TodoItem from '../TodoItem.vue'

describe('TodoItem 事件', () => {
  const todo = { id: 1, text: '学习测试', completed: false }

  it('勾选复选框触发 toggle 事件', async () => {
    const wrapper = mount(TodoItem, { props: { todo } })
    await wrapper.find('input[type="checkbox"]').trigger('change')

    expect(wrapper.emitted('toggle')).toBeTruthy()
    expect(wrapper.emitted('toggle')[0]).toEqual([1])
  })

  it('点击删除按钮触发 delete 事件', async () => {
    const wrapper = mount(TodoItem, { props: { todo } })
    await wrapper.find('button').trigger('click')

    expect(wrapper.emitted('delete')).toBeTruthy()
    expect(wrapper.emitted('delete')[0]).toEqual([1])
  })

  it('事件未触发时 emitted 返回 undefined', () => {
    const wrapper = mount(TodoItem, { props: { todo } })
    expect(wrapper.emitted('toggle')).toBeFalsy()
  })
})
```

## 二、具体用法

### 2.1 emitted() 返回结构

```js
// emitted() 返回对象，键为事件名，值为二维数组
// 每次触发是一个数组元素，数组内是触发时的参数
wrapper.emitted('click')         // [[args1], [args2], ...]
wrapper.emitted('click')[0]      // [args1] - 第一次触发的参数
wrapper.emitted('click')[0][0]   // args1[0] - 第一个参数
```

### 2.2 验证事件触发次数

```js
await button.trigger('click')
await button.trigger('click')
expect(wrapper.emitted('click')).toHaveLength(2)
```

### 2.3 验证复杂参数

```js
wrapper.emit('submit', { name: '张三', age: 25 })
expect(wrapper.emitted('submit')[0]).toEqual([{ name: '张三', age: 25 }])
```

## 三、注意事项与常见陷阱

- `emitted()` 返回的是**已触发事件**的记录，不是组件定义的 emits
- 不要测试 Vue 的 emit 机制本身，只测试事件是否在正确条件触发
- 多次触发事件时注意数组下标
