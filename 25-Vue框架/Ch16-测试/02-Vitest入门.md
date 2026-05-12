# Vitest 入门

## 一、概念说明

Vitest 是基于 Vite 构建的测试框架，天然支持 Vite 项目的配置（别名、插件等），运行速度极快，API 与 Jest 兼容。

```bash
# 安装
npm install -D vitest @vue/test-utils jsdom
```

```js
// vitest.config.ts
import { defineConfig } from 'vitest/config'

export default defineConfig({
  test: {
    environment: 'jsdom',         // 模拟浏览器环境
    globals: true,                // 全局 API（describe, it, expect）
    include: ['**/*.{test,spec}.{js,ts,jsx,tsx}'],
  }
})
```

```js
// tests/sum.test.ts
import { describe, it, expect } from 'vitest'

function sum(a, b) {
  return a + b
}

describe('sum', () => {
  it('1 + 2 = 3', () => {
    expect(sum(1, 2)).toBe(3)
  })

  it('负数相加', () => {
    expect(sum(-1, -2)).toBe(-3)
  })
})
```

## 二、具体用法

### 2.1 常用断言

```js
expect(value).toBe(42)            // 严格相等
expect(value).toEqual({ a: 1 })   // 深度相等
expect(value).toBeTruthy()        // 真值
expect(value).toContain('hello')  // 包含
expect(fn).toThrow()              // 抛出异常
expect(arr).toHaveLength(3)       // 长度
```

### 2.2 运行测试

```bash
npx vitest            # 监听模式
npx vitest run        # 单次运行
npx vitest --coverage # 带覆盖率
```

### 2.3 setup 文件

```js
// vitest.config.ts
export default defineConfig({
  test: {
    setupFiles: ['./tests/setup.ts']
  }
})
```

## 四、Mock 函数

```js
import { describe, it, expect, vi } from 'vitest'

describe('用户服务', () => {
  it('调用 API 获取用户', async () => {
    // 创建 mock 函数
    const fetchUser = vi.fn()
    fetchUser.mockResolvedValue({ id: 1, name: '张三' })

    const user = await fetchUser(1)
    expect(user.name).toBe('张三')
    expect(fetchUser).toHaveBeenCalledWith(1)
    expect(fetchUser).toHaveBeenCalledTimes(1)
  })

  it('模拟 API 失败', async () => {
    const fetchUser = vi.fn()
    fetchUser.mockRejectedValue(new Error('网络错误'))

    await expect(fetchUser(1)).rejects.toThrow('网络错误')
  })
})
```

## 五、Mock 模块

```js
import { describe, it, expect, vi } from 'vitest'

// Mock 整个模块
vi.mock('@/api/user', () => ({
  getUser: vi.fn().mockResolvedValue({ id: 1, name: '张三' }),
  updateUser: vi.fn().mockResolvedValue({ success: true })
}))

// Mock 部分模块
vi.mock('@/utils/helper', async (importOriginal) => {
  const mod = await importOriginal()
  return {
    ...mod,
    formatDate: vi.fn().mockReturnValue('2024-01-01')
  }
})
```

## 六、测试 Vue 组件

```js
import { describe, it, expect } from 'vitest'
import { mount } from '@vue/test-utils'
import Counter from '../components/Counter.vue'

describe('Counter', () => {
  it('初始值为0', () => {
    const wrapper = mount(Counter)
    expect(wrapper.text()).toContain('0')
  })

  it('点击按钮递增', async () => {
    const wrapper = mount(Counter)
    await wrapper.find('button').trigger('click')
    expect(wrapper.text()).toContain('1')
  })

  it('接受 initial prop', () => {
    const wrapper = mount(Counter, { props: { initial: 10 } })
    expect(wrapper.text()).toContain('10')
  })

  it('发出 change 事件', async () => {
    const wrapper = mount(Counter)
    await wrapper.find('button').trigger('click')
    expect(wrapper.emitted('change')).toHaveLength(1)
    expect(wrapper.emitted('change')[0]).toEqual([1])
  })
})
```

## 七、生命周期钩子

```js
import { describe, it, vi } from 'vitest'

describe('定时器测试', () => {
  it('延迟执行', () => {
    vi.useFakeTimers()

    const callback = vi.fn()
    setTimeout(callback, 1000)

    vi.advanceTimersByTime(500)
    expect(callback).not.toHaveBeenCalled()

    vi.advanceTimersByTime(500)
    expect(callback).toHaveBeenCalled()

    vi.useRealTimers()
  })
})
```

## 三、注意事项与常见陷阱

- 需要 `jsdom` 环境来测试 DOM 相关代码
- Vitest 支持 ESM，不需要额外配置 transform
- 使用 `vi.fn()` 创建模拟函数，`vi.spyOn()` 监视函数调用
- `vi.mock()` 的作用域是整个测试文件，放在文件顶部
- 每个测试用例应独立，使用 `beforeEach` 重置 mock 状态
