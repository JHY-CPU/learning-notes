# reactive 详解

## 一、概念说明

`reactive()` 为对象创建一个深层响应式代理。它使用 Proxy 包装对象，所有嵌套属性都是响应式的。与 ref 不同，reactive 的属性可以直接访问，无需 `.value`。

```vue
<script setup>
import { reactive } from 'vue'

const state = reactive({
  count: 0,
  user: {
    name: '张三',
    hobbies: ['编程', '阅读']
  }
})

// 直接访问，无需 .value
state.count++
state.user.name = '李四'
state.user.hobbies.push('音乐')
</script>

<template>
  <p>计数: {{ state.count }}</p>
  <p>用户: {{ state.user.name }}</p>
  <p>爱好: {{ state.user.hobbies.join(', ') }}</p>
  <button @click="state.count++">+1</button>
</template>
```

## 二、具体用法

### 2.1 基本用法

```vue
<script setup>
import { reactive } from 'vue'

const form = reactive({
  username: '',
  email: '',
  password: ''
})

const submit = () => {
  console.log('提交:', { ...form })
}
</script>

<template>
  <input v-model="form.username" placeholder="用户名" />
  <input v-model="form.email" placeholder="邮箱" />
  <input v-model="form.password" type="password" placeholder="密码" />
  <button @click="submit">注册</button>
</template>
```

### 2.2 数组操作

```vue
<script setup>
import { reactive } from 'vue'

const state = reactive({
  items: ['苹果', '香蕉']
})

// 以下操作都是响应式的
state.items.push('橙子')         // 添加
state.items.splice(0, 1)         // 删除
state.items[0] = '葡萄'          // 索引赋值
state.items.length = 0           // 清空
</script>
```

### 2.3 解构丢失响应式

```vue
<script setup>
import { reactive, toRefs } from 'vue'

const state = reactive({ count: 0, name: 'Vue' })

// 错误: 解构后丢失响应式
let { count, name } = state
count++ // 不会触发更新！

// 正确: 使用 toRefs
let { count: countRef, name: nameRef } = toRefs(state)
countRef.value++ // 保持响应式
</script>
```

## 三、注意事项与常见陷阱

- `reactive()` 只能用于对象类型（不能用于 string/number/boolean）
- 解构 reactive 对象会丢失响应式
- 替换整个对象（`state = newObj`）会丢失响应式
- `reactive()` 返回的代理与原始对象 `===` 不相等
- 对 reactive 对象添加新属性是响应式的
