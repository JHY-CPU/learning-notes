# 组合式API最佳实践

## 一、概念说明

遵循最佳实践能让组合式API代码更清晰、更可维护。主要包括命名规范、参数设计、返回值约定和文件组织等方面。

```vue
<script setup>
// ✅ 良好实践示例
import { useUserStore } from '@/stores/user'
import { useDebounceFn } from '@/composables/useDebounceFn'
import { useEventListener } from '@/composables/useEventListener'

const store = useUserStore()
const search = useDebounceFn(handleSearch, 300)
useEventListener(window, 'resize', handleResize)
</script>
```

## 二、具体用法

### 命名规范

```js
// ✅ 文件名：useXxx.js，函数名：useXxx()
// useMousePosition.js
export function useMousePosition() { /* ... */ }

// ✅ 返回值使用明确的命名
return { x, y, isMoving }  // 优于 ret1, ret2

// ✅ 选项参数使用对象，便于扩展
export function useFetch(url, options = {}) {
  const { immediate = true, timeout = 5000 } = options
}
```

### 参数设计

```js
// ✅ 支持ref参数
export function useTitle(newTitle) {
  const title = isRef(newTitle) ? newTitle : ref(newTitle)
  watch(title, (val) => { document.title = val })
  return title
}

// ✅ 提供默认值
export function useCounter(initialValue = 0, options = {}) {
  const { min = -Infinity, max = Infinity } = options
}
```

### 返回值约定

```js
// ✅ 返回ref而非reactive
return { count, doubled, increment }

// ✅ 返回值数组用于简单场景
return [isOpen, toggle]

// ✅ 返回函数用于以动作为主的场景
return useDebounce(search, 300)
```

### 错误处理

```js
export function useApi(url) {
  const error = ref(null)
  const data = ref(null)

  const execute = async () => {
    try {
      data.value = await fetch(url).then(r => r.json())
    } catch (e) {
      error.value = e
      // 不要静默吞掉错误
      console.error('[useApi]', e)
    }
  }

  return { data, error, execute }
}
```

## 三、注意事项与常见陷阱

1. **不要在循环/条件中调用**组合式函数（虽然Vue允许，但会创建意外的实例）
2. 组合式函数应该是**纯函数**（相同输入产生相同状态结构）
3. 避免在composables中操作DOM（除非是特定目的如useMouse）
4. 考虑TypeScript用户，提供完整的类型定义
5. 编写单元测试验证composable的行为
6. 文档化composable的用途、参数、返回值
