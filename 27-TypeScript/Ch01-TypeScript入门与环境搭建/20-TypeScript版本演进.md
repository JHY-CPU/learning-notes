# TypeScript 版本演进

## 一、概念说明

TypeScript 每隔数月发布新版本，引入语言特性和类型系统改进。了解版本演进有助于使用最新特性和理解旧代码。TypeScript 5.x 系列是当前主流版本，带来了许多重要的类型系统增强。

## 二、具体用法

### 2.1 TypeScript 4.x 重要特性

```typescript
// TS 4.1: 模板字面量类型
type EventName = `on${Capitalize<"click" | "focus" | "blur">}`;
// 结果: "onClick" | "onFocus" | "onBlur"

// TS 4.4: catch 中的 unknown
try {
  throw new Error("出错了");
} catch (error) {
  // strict 模式下 error 类型为 unknown
  if (error instanceof Error) {
    console.log(error.message);
  }
}
// 输出: 出错了

// TS 4.9: satisfies 运算符
type ColorFormat = "hex" | "rgb" | "hsl";
const colors = {
  primary: "#007bff",
  danger: "#dc3545",
} satisfies Record<ColorFormat, string>;
// colors 保留精确类型，同时检查满足约束
```

### 2.2 TypeScript 5.x 新特性

```typescript
// TS 5.0: const 类型参数
function fn<const T extends readonly string[]>(args: T): T {
  return args;
}
const result = fn(["a", "b"]); // 类型: readonly ["a", "b"]

// TS 5.0: 装饰器（ECMAScript 标准）
function log(target: any, context: ClassMethodDecoratorContext) {
  return function (this: any, ...args: any[]) {
    console.log(`调用: ${String(context.name)}`);
    return target.apply(this, args);
  };
}

class MyClass {
  @log
  greet(name: string) {
    return `Hello, ${name}`;
  }
}

const obj = new MyClass();
obj.greet("World");
// 输出:
// 调用: greet
// (方法返回 "Hello, World")
```

### 2.3 版本对应关系

```
TypeScript 5.0 (2023.3)  - const 类型参数、装饰器
TypeScript 5.1 (2023.6)  - getter/setter 类型不同
TypeScript 5.2 (2023.8)  - using 声明（显式资源管理）
TypeScript 5.3 (2023.11) - import 属性
TypeScript 5.4 (2024.3)  - NoInfer 工具类型
TypeScript 5.5 (2024.6)  - 类型推断改进
TypeScript 5.6 (2024.9)  - 更多类型收窄改进
TypeScript 5.7 (2024.12) - 相对路径导入改进
```

## 三、注意事项与常见陷阱

1. **版本兼容性**：新特性需要对应的 TypeScript 版本才能使用
2. **Node.js 最低版本**：TS 5.x 要求 Node.js 12+，部分特性需要更高版本
3. **不要过度追求新特性**：考虑团队和依赖库的版本兼容性
4. **Breaking Changes**：大版本升级时注意阅读官方迁移指南
