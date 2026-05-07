# 模拟 Mock

## 一、概念说明

Mock（模拟）用于替换测试中的真实依赖，隔离被测试代码。Vitest 内置了强大的 mock 功能，支持 `vi.fn()`、`vi.mock()`、`vi.spyOn()` 三种核心 API。

```js
import { vi, describe, it, expect } from 'vitest'

describe('Mock 示例', () => {
  it('vi.fn() 创建模拟函数', () => {
    const mockFn = vi.fn()
    mockFn('hello')
    mockFn('world')

    expect(mockFn).toHaveBeenCalled()
    expect(mockFn).toHaveBeenCalledTimes(2)
    expect(mockFn).toHaveBeenCalledWith('hello')
  })

  it('vi.fn() 带实现', () => {
    const mockFn = vi.fn((a, b) => a + b)
    expect(mockFn(1, 2)).toBe(3)
  })

  it('vi.fn() 模拟返回值', () => {
    const mockFn = vi.fn()
    mockFn.mockReturnValue(42)
    expect(mockFn()).toBe(42)

    mockFn.mockResolvedValue(100) // Promise.resolve(100)
  })
})
```

## 二、具体用法

### 2.1 vi.mock() 模拟模块

```js
// 模拟整个 axios 模块
vi.mock('axios', () => ({
  default: {
    get: vi.fn().mockResolvedValue({ data: [] }),
    post: vi.fn().mockResolvedValue({ data: { id: 1 } }),
    create: vi.fn(() => ({
      get: vi.fn(),
      post: vi.fn(),
      interceptors: { request: { use: vi.fn() }, response: { use: vi.fn() } }
    }))
  }
}))
```

### 2.2 vi.spyOn() 监视函数

```js
it('调用了 console.log', () => {
  const spy = vi.spyOn(console, 'log')
  console.log('test')
  expect(spy).toHaveBeenCalledWith('test')
  spy.mockRestore() // 恢复原始函数
})
```

### 2.3 mock 模拟模块的部分导出

```js
// 只 mock 模块的部分导出
vi.mock('../utils/helper', async () => {
  const actual = await vi.importActual('../utils/helper')
  return { ...actual, formatDate: vi.fn(() => '2024-01-01') }
})
```

### 2.4 自动清理

```js
// vitest.config.ts
export default defineConfig({
  test: {
    mockReset: true,    // 每个测试后重置 mock
    restoreMocks: true  // 每个测试后恢复原始实现
  }
})
```

## 三、注意事项与常见陷阱

- `vi.mock` 必须在文件顶部调用（会被提升）
- 使用 `vi.spyOn` 后记得 `mockRestore`，避免影响其他测试
- Mock 要尽量接近真实接口，避免 mock 过于简化导致测试通过但实际出错
