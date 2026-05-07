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

## 三、注意事项与常见陷阱

- 需要 `jsdom` 环境来测试 DOM 相关代码
- Vitest 支持 ESM，不需要额外配置 transform
- 使用 `vi.fn()` 创建模拟函数，`vi.spyOn()` 监视函数调用
