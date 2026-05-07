# 测试配置Vitest

## 一、概念说明

Vitest 是基于 Vite 的测试框架，原生支持 TypeScript，无需额外配置。它的 API 与 Jest 兼容，但执行速度更快，配置更简单。

## 二、具体用法

### 2.1 安装与配置

```bash
npm install -D vitest @vitest/coverage-v8
```

```typescript
// vitest.config.ts
import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    globals: true,
    environment: 'jsdom',
    include: ['src/**/*.{test,spec}.ts'],
    coverage: {
      provider: 'v8',
      reporter: ['text', 'json', 'html'],
      include: ['src/**/*.ts'],
      exclude: ['src/**/*.d.ts', 'src/**/*.test.ts'],
    },
  },
});
```

### 2.2 基本测试

```typescript
import { describe, it, expect, vi } from 'vitest';

describe('计算器', () => {
  it('加法', () => {
    expect(1 + 1).toBe(2);
  });

  it('异步', async () => {
    const result = await Promise.resolve(42);
    expect(result).toBe(42);
  });

  it('mock', () => {
    const fn = vi.fn().mockReturnValue(42);
    expect(fn()).toBe(42);
  });
});
```

### 2.3 包含类型测试

```typescript
import { expectTypeOf, describe, it } from 'vitest';

describe('类型测试', () => {
  it('应该有正确的类型', () => {
    expectTypeOf<string>().toEqualTypeOf<string>();
    expectTypeOf(42).toBeNumber();
    expectTypeOf({ name: '张三' }).toMatchObjectType<{ name: string }>();
  });
});
```

### 2.4 环境配置

```typescript
// vitest.config.ts
export default defineConfig({
  test: {
    // 不同文件用不同环境
    environmentMatchGlobs: [
      ['**/*.dom.test.ts', 'jsdom'],
      ['**/*.node.test.ts', 'node'],
    ],
  },
});
```

### 2.5 Package.json 脚本

```json
{
  "scripts": {
    "test": "vitest",
    "test:run": "vitest run",
    "test:coverage": "vitest run --coverage",
    "test:watch": "vitest --watch"
  }
}
```

## 三、注意事项与常见陷阱

1. **Vitest 原生支持 TS**：不需要 ts-jest 或 babel
2. **`globals: true` 避免导入**：`describe`、`it`、`expect` 全局可用
3. **环境配置**：DOM 测试用 `jsdom`，Node 测试用 `node`
4. **coverage provider 用 `v8`**：比 `istanbul` 更快
5. **测试文件放在 `src/` 中**：确保被 tsconfig 覆盖
