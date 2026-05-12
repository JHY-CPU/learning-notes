# toRaw 获取原始对象

## 一、概念说明

`toRaw()` 返回 reactive 或 readonly 代理的原始对象。常用于临时跳过响应式系统、与外部库交互、或调试时查看原始数据。

```vue
<script setup>
import { reactive, toRaw } from 'vue'

const state = reactive({ count: 0, name: 'Vue' })

const raw = toRaw(state)
console.log(raw === state) // false（原始对象 !== 代理）

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
</script>
```

### 2.4 与 markRaw 对比

```js
import { reactive, toRaw, markRaw } from 'vue'

// toRaw: 从代理中提取原始对象（响应式 → 原始）
const proxy = reactive({ data: 1 })
const raw = toRaw(proxy)  // 原始对象

// markRaw: 标记对象不被转为代理（原始 → 不可代理）
const obj = markRaw({ data: 1 })
const proxy2 = reactive(obj)  // obj 不会被转为代理
console.log(proxy2 === obj)   // true
```

### 2.5 Map/Set 的原始对象

```vue
<script setup>
import { reactive, toRaw } from 'vue'

const state = reactive({
  map: new Map([['key', 'value']]),
  set: new Set([1, 2, 3])
})

// 获取原始 Map/Set
const rawMap = toRaw(state.map)
const rawSet = toRaw(state.set)

rawMap.set('newKey', 'newValue') // 不触发响应式
</script>
```

## 三、常见用例

| 场景 | 说明 |
|------|------|
| 传递数据给第三方库 | 避免 Proxy 干扰第三方库 |
| 调试响应式数据 | 查看原始值而非代理 |
| 性能敏感操作 | 跳过响应式追踪 |
| 与外部系统交互 | 序列化/比较等操作 |

## 四、注意事项与常见陷阱

- `toRaw()` 只返回最外层代理的原始对象
- 修改 `toRaw()` 返回的原始对象不会触发视图更新
- 对非代理对象调用 `toRaw()` 返回对象本身
- `toRaw()` 对 ref 无效，需要 `toRaw(ref.value)`
- 不要在正常开发中频繁使用 toRaw，它跳过了响应式系统
- toRaw 返回的原始对象修改后，代理对象也会反映变化（它们是同一个引用）
- markRaw 是永久标记，toRaw 是临时提取
