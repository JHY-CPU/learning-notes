# ref与reactive在setup中

## 一、概念说明

`ref`和`reactive`是创建响应式数据的两种方式：
- **ref**：可包装任意类型值（原始值、对象），通过`.value`访问
- **reactive**：只能包装对象类型，直接访问属性

```vue
<template>
  <div>
    <p>ref计数: {{ count }}</p>
    <p>reactive用户: {{ user.name }} - {{ user.age }}岁</p>
    <button @click="count++">ref+1</button>
    <button @click="user.age++">长一岁</button>
  </div>
</template>

<script setup>
import { ref, reactive } from 'vue'

const count = ref(0)
const user = reactive({ name: '张三', age: 20 })
</script>
```

## 二、具体用法

### ref的使用

```js
import { ref } from 'vue'

const num = ref(10)
console.log(num.value) // 10
num.value = 20         // 修改值

const arr = ref([1, 2, 3])
arr.value.push(4)      // 数组操作

const obj = ref({ x: 1 })
obj.value.x = 2        // 嵌套属性
```

### reactive的使用

```js
import { reactive } from 'vue'

const state = reactive({
  name: '李四',
  scores: { math: 90, english: 85 },
  hobbies: ['读书', '跑步']
})

state.name = '王五'           // 直接修改
state.scores.math = 95       // 嵌套对象也是响应式的
state.hobbies.push('编程')    // 数组方法可用
```

### ref vs reactive 选择

| 特性 | ref | reactive |
|------|-----|----------|
| 支持原始值 | 是 | 否 |
| 需要.value | 是（JS中） | 否 |
| 可整体替换 | 是 | 是（需Object.assign） |
| 解构保持响应式 | 否 | 否 |
| 适用场景 | 通用 | 复杂对象状态 |

## 三、注意事项与常见陷阱

1. **ref在模板中自动解包**：无需`.value`，直接写`{{ count }}`
2. **ref在reactive中自动解包**：`reactive({ count: ref(0) })`中`state.count`无需`.value`
3. **reactive不能整体替换**：`state = newObj`会丢失响应式，用`Object.assign(state, newObj)`
4. **解构会丢失响应式**：`const { name } = reactive(...)`后`name`不再是响应式的
5. 推荐：对原始值用`ref`，复杂对象两种均可，团队统一风格
