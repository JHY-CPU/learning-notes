# 测试配置Jest

## 一、概念说明

Jest 配合 `ts-jest` 可以直接运行 TypeScript 测试。虽然 Vitest 是更现代的选择（原生 ESM、更快），但 Jest 仍然是很多存量项目的测试框架。关键配置包括 `ts-jest` 预设、路径映射和类型检查选项。

## 二、具体用法

### 2.1 安装依赖

```bash
npm install -D jest ts-jest @types/jest
```

### 2.2 基础配置

```typescript
// jest.config.ts
import type { Config } from "jest";

const config: Config = {
  preset: "ts-jest",
  testEnvironment: "node",
  roots: ["<rootDir>/src"],
  testMatch: ["**/*.test.ts", "**/*.spec.ts"],
  moduleNameMapper: {
    "^@/(.*)$": "<rootDir>/src/$1",
  },
  collectCoverageFrom: [
    "src/**/*.ts",
    "!src/**/*.d.ts",
    "!src/**/*.test.ts",
    "!src/**/index.ts",
  ],
};

export default config;
```

### 2.3 ts-jest 高级配置

```typescript
// jest.config.ts
const config: Config = {
  preset: "ts-jest",
  transform: {
    "^.+\\.tsx?$": [
      "ts-jest",
      {
        tsconfig: "tsconfig.json",
        diagnostics: true,        // 启用类型检查
        isolatedModules: true,    // 加速编译
      },
    ],
  },
  // 支持 ES Module 模拟
  moduleNameMapper: {
    "^@/(.*)$": "<rootDir>/src/$1",
    "^~/(.*)$": "<rootDir>/$1",
  },
};

export default config;
```

### 2.4 编写类型安全的测试

```typescript
// math.ts
export function add(a: number, b: number): number {
  return a + b;
}

export function divide(a: number, b: number): number {
  if (b === 0) throw new Error("除数不能为零");
  return a / b;
}

// math.test.ts
import { add, divide } from "./math";

describe("math", () => {
  describe("add", () => {
    it("应该正确相加两个数", () => {
      expect(add(1, 2)).toBe(3);
      expect(add(-1, 1)).toBe(0);
    });
  });

  describe("divide", () => {
    it("应该正确相除", () => {
      expect(divide(10, 2)).toBe(5);
    });

    it("除以零应该抛出错误", () => {
      expect(() => divide(10, 0)).toThrow("除数不能为零");
    });
  });
});
```

### 2.5 Mock 与类型

```typescript
// 模拟模块
jest.mock("./api", () => ({
  fetchUser: jest.fn().mockResolvedValue({ id: 1, name: "Alice" }),
}));

// 类型安全的 mock
const mockFetchUser = jest.fn<Promise<{ id: number; name: string }>, [number]>();
mockFetchUser.mockResolvedValue({ id: 1, name: "Alice" });

// Mock 类型辅助
type MockedFunction<T extends (...args: any[]) => any> = jest.MockedFunction<T>;
```

### 2.6 Package.json 脚本

```json
{
  "scripts": {
    "test": "jest",
    "test:watch": "jest --watch",
    "test:coverage": "jest --coverage",
    "test:ci": "jest --ci --coverage --watchAll=false"
  }
}
```

### 2.7 与 JavaScript 的对比

```javascript
// JavaScript Jest：无类型检查
const { add } = require("./math");
test("adds", () => {
  expect(add(1, "2")).toBe(3); // 运行时可能出错
});

// TypeScript + ts-jest：编译时类型检查
import { add } from "./math";
test("adds", () => {
  expect(add(1, "2")).toBe(3); // 编译错误：string 不能赋给 number
});
```

## 三、注意事项与常见陷阱

1. **`ts-jest` 需要 TypeScript**：确保已安装 `typescript`
2. **Jest 不原生支持 ESM**：需要额外配置 `transform` 或使用 Vitest
3. **路径映射需 `moduleNameMapper`**：与 `tsconfig.json` 的 `paths` 对应
4. **Jest 比 Vitest 慢**：新项目推荐 Vitest（更快、原生 ESM）
5. **类型测试用 `expect-type`**：Jest 不原生支持类型断言，可用第三方库
6. **`isolatedModules: true`**：加速 ts-jest 的编译速度
