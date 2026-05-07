# 响应式原理 - Proxy

## 一、概念说明

Vue 3 使用 ES6 的 **Proxy** 和 **Reflect** 实现响应式。Proxy 可以拦截对象的各种操作（读取、设置、删除等），在拦截时实现依赖收集和触发更新。相比 Vue 2 的 `Object.defineProperty`，Proxy 能拦截更多操作且性能更好。

```vue
<script setup>
// Proxy 响应式简化原理
// Vue 3 内部实现（简化版）
function reactive(target) {
  return new Proxy(target, {
    get(target, key, receiver) {
      track(target, key) // 依赖收集
      return Reflect.get(target, key, receiver)
    },
    set(target, key, value, receiver) {
      const result = Reflect.set(target, key, value, receiver)
      trigger(target, key) // 触发更新
      return result
    },
    deleteProperty(target, key) {
      const result = Reflect.deleteProperty(target, key)
      trigger(target, key)
      return result
    }
  })
}
</script>
```

## 二、具体用法

### 2.1 Proxy 基本示例

```js
const original = { count: 0, name: 'Vue' }

const proxy = new Proxy(original, {
  get(target, key) {
    console.log(`读取: ${key}`)
    return Reflect.get(target, key)
  },
  set(target, key, value) {
    console.log(`设置: ${key} = ${value}`)
    return Reflect.set(target, key, value)
  }
})

proxy.count      // 读取: count
proxy.count = 1  // 设置: count = 1
```

### 2.2 依赖收集与触发更新

```js
// 简化的响应式系统
const targetMap = new WeakMap()

function track(target, key) {
  if (activeEffect) {
    let depsMap = targetMap.get(target)
    if (!depsMap) {
      depsMap = new Map()
      targetMap.set(target, depsMap)
    }
    let dep = depsMap.get(key)
    if (!dep) {
      dep = new Set()
      depsMap.set(key, dep)
    }
    dep.add(activeEffect)
  }
}

function trigger(target, key) {
  const depsMap = targetMap.get(target)
  if (!depsMap) return
  const dep = depsMap.get(key)
  if (dep) {
    dep.forEach(effect => effect())
  }
}
```

### 2.3 Proxy 能做到而 defineProperty 不能的

```js
// 1. 新增属性
const obj = reactive({})
obj.newProp = 'value' // 响应式

// 2. 删除属性
delete obj.newProp // 响应式

// 3. 数组索引
const arr = reactive([])
arr[0] = 'value' // 响应式

// 4. 数组 length
arr.length = 0 // 响应式
```

## 三、注意事项与常见陷阱

- Proxy 无法被 polyfill，因此 Vue 3 不支持 IE11
- Proxy 包装后的对象与原始对象 `===` 不相等
- 使用 `toRaw()` 可以获取 Proxy 后面的原始对象
- Proxy 只能代理引用类型，基本类型用 ref 包装
- `reactive()` 对同一对象多次调用返回同一个代理
