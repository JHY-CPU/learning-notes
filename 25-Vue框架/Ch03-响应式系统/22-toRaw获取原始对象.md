# toRaw 获取原始对象

## 一、概念说明

`toRaw()` 返回 reactive 或 readonly 代理的原始对象。常用于临时跳过响应式系统、与外部库交互、或调试时查看原始数据。

```vue
<script setup>
import { reactive, toRaw } from 'vue'

const state = reactive({ count: 0, name: 'Vue' })

// 获取原始对象
const raw = toRaw(state)
console.log(raw === state) // false（原始对象 !== 代理）

// 修改原始对象不会触发响应式更新
raw.count = 100
console.log(state.count) // 100（但视图不会更新）
</script>
```

## 二、具体用法

### 2.1 跳过响应式传递

```vue
<script setup>
import { reactive, toRaw } from 'vue'

const form = reactive({
  username: '',
  email: ''
})

function submitForm() {
  // 发送原始对象给第三方库（不需要响应式）
  const rawData = toRaw(form)
  externalApi.send(rawData)
}
</script>
```

### 2.2 调试原始数据

```vue
<script setup>
import { reactive, toRaw } from 'vue'

const state = reactive({
  items: [{ id: 1, name: '项目1' }]
})

function debugState() {
  // 打印原始对象，避免 Proxy 的干扰
  console.log(JSON.stringify(toRaw(state), null, 2))
}
</script>
```

### 2.3 与 ref 配合

```vue
<script setup>
import { ref, toRaw } from 'vue'

const state = ref({ count: 0 })

// ref 需要先取 .value
const raw = toRaw(state.value)
```

## 三、注意事项与常见陷阱

- `toRaw()` 只返回最外层代理的原始对象
- 修改 `toRaw()` 返回的原始对象不会触发视图更新
- 对非代理对象调用 `toRaw()` 返回对象本身
- `toRaw()` 对 ref 无效，需要 `toRaw(ref.value)`
- 不要在正常开发中频繁使用 toRaw，它跳过了响应式系统
