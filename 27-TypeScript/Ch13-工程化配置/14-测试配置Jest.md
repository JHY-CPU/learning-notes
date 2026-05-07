# 测试配置Jest

## 一、概念说明

Jest 配合 `ts-jest` 可以运行 TypeScript 测试。虽然 Vitest 是更现代的选择，但 Jest 仍然是很多项目的测试框架。

## 二、具体用法

### 2.1 安装

```bash
npm install -D jest ts-jest @types/jest
```

### 2.2 配置

```typescript
// jest.config.ts
import type { Config } from 'jest';

const config: Config = {
  preset: 'ts-jest',
  testEnvironment: 'node',
  roots: ['<rootDir>/src'],
  testMatch: ['**/*.test.ts'],
  moduleNameMapper: {
    '^@/(.*)$': '<rootDir>/src/$1',
  },
  collectCoverageFrom: [
    'src/**/*.ts',
    '!src/**/*.d.ts',
    '!src/**/*.test.ts',
  ],
};

export default config;
```

### 2.3 ts-jest 配置

```typescript
// jest.config.ts
const config: Config = {
  preset: 'ts-jest',
  transform: {
    '^.+\\.tsx?$': ['ts-jest', {
      tsconfig: 'tsconfig.json',
      diagnostics: true, // 类型检查
    }],
  },
};

export default config;
```

### 2.4 基本测试

```typescript
import { myFunction } from './myModule';

describe('myModule', () => {
  it('应该返回正确结果', () => {
    const result = myFunction(1, 2);
    expect(result).toBe(3);
  });

  it('mock 应该工作', () => {
    const fn = jest.fn().mockReturnValue(42);
    expect(fn()).toBe(42);
  });
});
```

### 2.5 Package.json

```json
{
  "scripts": {
    "test": "jest",
    "test:watch": "jest --watch",
    "test:coverage": "jest --coverage"
  }
}
```

## 三、注意事项与常见陷阱

1. **`ts-jest` 需要 TypeScript**：确保已安装
2. **Jest 不理解 ESM**：需要额外配置或使用 Vitest
3. **路径映射需要 `moduleNameMapper`**：与 tsconfig 的 paths 对应
4. **Jest 比 Vitest 慢**：新项目推荐 Vitest
5. **类型测试用 `expect-type` 或 `vitest`**：Jest 不原生支持
