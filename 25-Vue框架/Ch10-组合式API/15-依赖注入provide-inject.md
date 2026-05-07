# 依赖注入provide/inject

## 一、概念说明

`provide`和`inject`实现**跨层级**组件通信，祖先组件提供数据，后代组件注入使用。它避免了prop逐层传递的问题（"prop drilling"）。

```vue
<!-- 祖先组件 -->
<script setup>
import { provide, ref } from 'vue'

const theme = ref('dark')
provide('theme', theme)
provide('version', '1.0.0')
</script>

<!-- 任意后代组件 -->
<script setup>
import { inject } from 'vue'

const theme = inject('theme')       // 响应式ref
const version = inject('version')   // 静态值
</script>
```

## 二、具体用法

### 基础用法

```vue
<!-- Parent.vue -->
<script setup>
import { provide, ref, readonly } from 'vue'

const count = ref(0)
const user = ref({ name: '张三' })

provide('count', count)           // 提供响应式ref
provide('user', readonly(user))   // 只读保护
provide('increment', () => count.value++) // 提供方法
</script>

<!-- Child.vue (任意深度的后代) -->
<script setup>
import { inject } from 'vue'

const count = inject('count')
const user = inject('user')
const increment = inject('increment')
</script>
```

### 带默认值的注入

```vue
<script setup>
import { inject } from 'vue'

// 第二个参数是默认值
const theme = inject('theme', 'light')
const config = inject('config', () => ({ debug: false }), true)
// 第三个参数true表示默认值是工厂函数
</script>
```

### 使用Symbol作为Key

```js
// keys.js
export const THEME_KEY = Symbol('theme')
export const USER_KEY = Symbol('user')
```

```vue
<script setup>
import { provide } from 'vue'
import { THEME_KEY } from './keys'

provide(THEME_KEY, 'dark')
</script>
```

## 三、注意事项与常见陷阱

1. `provide`/`inject`绑定**不是响应式的**，除非传入ref/reactive对象
2. 注入的数据应视为**只读**，不要直接修改，通过provide的方法修改
3. 使用Symbol作为key可避免命名冲突
4. 如果key在祖先中没有provide，inject返回`undefined`或默认值
5. 适用于深层嵌套组件、插件开发、组件库封装
