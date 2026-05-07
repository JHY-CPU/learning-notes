# TypeScript 5.0概述

## 一、概念说明

TypeScript 5.0 是一个重大版本，带来了 TC39 Stage 3 装饰器、`const` 类型参数、模块解析改进等重要特性。这些特性使 TypeScript 更接近 ECMAScript 标准。

## 二、具体用法

### 2.1 主要新特性一览

```typescript
// 1. TC39 Stage 3 装饰器
function log(target: any, context: ClassMethodDecoratorContext) {
  return function (this: any, ...args: any[]) {
    console.log(`调用: ${String(context.name)}`);
    return target.apply(this, args);
  };
}

// 2. const 类型参数
function identity<const T>(value: T): T {
  return value;
}
const result = identity({ x: 1, y: 'hello' });
// result 的类型是 { readonly x: 1; readonly y: 'hello' }

// 3. 所有枚举都成为联合枚举
enum Direction { Up, Down, Left, Right }
// Direction 现在是 0 | 1 | 2 | 3 的联合

// 4. --moduleResolution bundler
// 匹配 Vite/Webpack 等打包工具的解析行为
```

### 2.2 装饰器（全新实现）

```typescript
// 不再需要 experimentalDecorators
function sealed(target: typeof MyClass, context: ClassDecoratorContext) {
  Object.seal(target.prototype);
  Object.seal(target);
}

@sealed
class MyClass {
  name = 'test';
}
```

### 2.3 const 类型参数

```typescript
// 没有 const — 类型被拓宽
function arr<T>(value: T[]): T[] { return value; }
const a = arr([1, 2, 3]); // number[]

// 有 const — 保留字面量类型
function constArr<const T>(value: T[]): T[] { return value; }
const b = constArr([1, 2, 3]); // [1, 2, 3]
```

### 2.4 模块解析 bundler

```json
{
  "compilerOptions": {
    "moduleResolution": "bundler"
  }
}
```

## 三、注意事项与常见陷阱

1. **Stage 3 装饰器与旧版不兼容**：需要移除 `experimentalDecorators`
2. **`const` 类型参数适合配置对象**：避免不必要的类型拓宽
3. **bundler 解析策略**：推荐用于 Vite/Webpack 项目
4. **枚举变为联合枚举**：类型检查更精确
5. **TypeScript 5.0 要求 Node.js 12.20+**
