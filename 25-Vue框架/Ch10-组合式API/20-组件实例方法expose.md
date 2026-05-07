# 组件实例方法expose

## 一、概念说明

`expose`是`setup()`函数的第三个参数（context.expose），用于在非`<script setup>`的setup函数中控制暴露给父组件的内容。

```vue
<!-- 子组件 -->
<script>
import { ref } from 'vue'

export default {
  setup(props, { expose }) {
    const count = ref(0)
    const name = ref('内部')

    const increment = () => count.value++
    const reset = () => { count.value = 0 }

    // 只暴露increment和reset，不暴露count和name
    expose({ increment, reset })

    return { count, name }
  }
}
</script>
```

```vue
<!-- 父组件 -->
<template>
  <Child ref="childRef" />
  <button @click="childRef?.increment()">调用子组件方法</button>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import Child from './Child.vue'

const childRef = ref(null)

onMounted(() => {
  // 只能访问expose暴露的内容
  childRef.value?.increment()  // OK
  // childRef.value.count       // 未暴露，不可访问
})
</script>
```

## 二、具体用法

### 对比：有无expose

```vue
<!-- 无expose：setup返回的所有内容暴露 -->
<script>
export default {
  setup() {
    const count = ref(0)
    const secret = ref('密码')
    return { count, secret }  // 父组件都能访问
  }
}
</script>

<!-- 有expose：只暴露指定内容 -->
<script>
export default {
  setup(props, { expose }) {
    const count = ref(0)
    const secret = ref('密码')
    expose({ count })  // 父组件只能访问count
    return { count, secret }
  }
}
</script>
```

### 表单组件暴露验证方法

```vue
<script>
import { ref } from 'vue'

export default {
  setup(props, { expose }) {
    const fields = ref({ username: '', password: '' })

    const validate = () => {
      return fields.value.username && fields.value.password.length >= 6
    }

    const getValues = () => ({ ...fields.value })

    expose({ validate, getValues })

    return { fields }
  }
}
</script>
```

## 三、注意事项与常见陷阱

1. 调用`expose()`后，只有显式暴露的内容父组件才能访问
2. `<script setup>`使用`defineExpose`，普通setup使用`context.expose`
3. 不调用expose时，setup返回值默认全部暴露
4. 暴露方法比暴露状态更好，保持封装性
5. 异步组件加载完成前，ref.value为null，需要做空值判断
