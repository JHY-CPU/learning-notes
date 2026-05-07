# 测试 Props

## 一、概念说明

测试组件的 Props 接收和渲染是否正确，包括验证默认值、类型检查、动态更新等行为。

```vue
<!-- Alert.vue -->
<script setup>
const props = defineProps({
  type: { type: String, default: 'info', validator: v => ['info', 'success', 'warning', 'error'].includes(v) },
  message: { type: String, required: true },
  closable: { type: Boolean, default: false }
})
const emit = defineEmits(['close'])
</script>

<template>
  <div :class="['alert', `alert-${type}`]" role="alert">
    {{ message }}
    <button v-if="closable" @click="emit('close')">关闭</button>
  </div>
</template>
```

```js
import { mount } from '@vue/test-utils'
import Alert from '../Alert.vue'

describe('Alert Props', () => {
  it('必需的 message prop', () => {
    const wrapper = mount(Alert, { props: { message: '提示信息' } })
    expect(wrapper.text()).toContain('提示信息')
  })

  it('默认 type 为 info', () => {
    const wrapper = mount(Alert, { props: { message: '提示' } })
    expect(wrapper.classes()).toContain('alert-info')
  })

  it('type=success 添加对应类名', () => {
    const wrapper = mount(Alert, { props: { message: '成功', type: 'success' } })
    expect(wrapper.classes()).toContain('alert-success')
  })

  it('closable=true 显示关闭按钮', () => {
    const wrapper = mount(Alert, { props: { message: '可关闭', closable: true } })
    expect(wrapper.find('button').exists()).toBe(true)
  })

  it('动态更新 props', async () => {
    const wrapper = mount(Alert, { props: { message: '原始' } })
    await wrapper.setProps({ message: '更新后' })
    expect(wrapper.text()).toContain('更新后')
  })
})
```

## 二、具体用法

### 2.1 更新 Props

```js
const wrapper = mount(Comp, { props: { count: 0 } })
await wrapper.setProps({ count: 10 })
expect(wrapper.text()).toContain('10')
```

### 2.2 测试缺少必需 Props

```js
// 缺少 required prop 会在控制台产生警告
const spy = vi.spyOn(console, 'warn').mockImplementation(() => {})
mount(Alert, { props: {} }) // 缺少 message
expect(spy).toHaveBeenCalled()
```

## 三、注意事项与常见陷阱

- `setProps` 后需要 `await` 等待响应式更新
- 测试时应覆盖 prop 的默认值、边界值和动态更新
- 不要测试 Vue 本身的 prop 验证机制，只测试组件行为
