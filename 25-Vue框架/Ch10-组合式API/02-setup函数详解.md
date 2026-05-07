# setup()函数详解

## 一、概念说明

`setup()`是组合式API的入口函数，在组件创建**之前**执行（beforeCreate之前）。它是组件中使用组合式API的核心场所。

```vue
<script>
import { ref } from 'vue'

export default {
  props: ['title'],
  emits: ['update'],
  setup(props, context) {
    // props 是响应式的，不能解构
    console.log(props.title)

    // context 包含 attrs、slots、emit、expose
    const count = ref(0)
    const increment = () => count.value++

    // 返回值暴露给模板
    return { count, increment }
  }
}
</script>
```

## 二、具体用法

### setup的执行时机
- 在`beforeCreate`之前调用
- 此时组件实例尚未创建，`this`为`undefined`

### setup的两个参数

```vue
<script>
export default {
  props: ['msg'],
  emits: ['change'],
  setup(props, { attrs, slots, emit, expose }) {
    // props: 响应式的props对象（不可解构，会丢失响应式）
    console.log(props.msg)

    // emit: 触发事件
    const handleClick = () => emit('change', '新值')

    // expose: 暴露组件方法给父组件
    const reset = () => { /* ... */ }
    expose({ reset })

    return { handleClick }
  }
}
</script>
```

### context对象的四个属性
| 属性 | 说明 |
|------|------|
| `attrs` | 非props的attribute |
| `slots` | 插槽内容 |
| `emit` | 触发自定义事件 |
| `expose` | 暴露公共方法 |

## 三、注意事项与常见陷阱

1. **不能对props解构**：`const { title } = props` 会丢失响应式，使用 `toRefs(props)` 代替
2. `setup()`不能是async函数（除非配合`<Suspense>`）
3. `setup()`中没有`this`，无法访问组件实例
4. 推荐使用`<script setup>`语法糖简化写法
5. `setup()`返回的对象/函数中的属性在模板中可直接使用
