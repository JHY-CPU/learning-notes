# TypeScript 5.1新特性

## 一、概念说明

TypeScript 5.1 带来了 getter/setter 类型解耦、尾部元组元素简化、JSX 支持改进等特性。其中 getter 和 setter 可以拥有不同的类型是重要的改进，使得属性的读写可以有不同语义。

## 二、具体用法

### 2.1 getter/setter 类型解耦

```typescript
// 5.1 之前：getter 和 setter 必须有完全相同的类型
class OldBox {
  private _value: string | number = "";

  get value(): string | number {
    return this._value;
  }

  set value(v: string | number) {
    this._value = v;
  }
}

// 5.1+：getter 和 setter 可以有不同类型
// 只要 setter 参数类型是 getter 返回类型的子类型
class SmartBox {
  private _value = "";

  get value(): string {
    return this._value;
  }

  set value(v: string | number) {
    this._value = String(v);
  }
}

const box = new SmartBox();
box.value = 42;             // OK — set 接受 number
const v: string = box.value; // OK — get 返回 string
```

### 2.2 尾部元组元素简化

```typescript
// 5.1 之前：可变长度元组较繁琐
type OldTuple = [string, ...(number | string)[]];

// 5.1+：更简洁的写法
type FlexTuple = [string, ...number[]];

const t1: FlexTuple = ["hello"];
const t2: FlexTuple = ["hello", 1, 2, 3];
// const t3: FlexTuple = ["hello", 1, "a"]; // 错误
```

### 2.3 JSX 支持改进

```typescript
// 5.1+：支持 JSX 中的解构默认值
// <Component { name = '默认名' } />

// 无额外 import 的 JSX
// 不需要 `import React`（配置 jsx: "react-jsx"）
const element = <div>Hello</div>;
```

### 2.4 更好的 `Object.keys` 类型

```typescript
// 5.1+：Object.keys 返回更精确的类型
interface Point {
  x: number;
  y: number;
}

function logKeys(p: Point) {
  const keys = Object.keys(p); // (keyof Point)[] 在某些上下文中
  // 仍然是 string[]，但工具函数可以利用
}
```

### 2.5 实际应用：配置类

```typescript
class Config {
  private _data: Record<string, unknown> = {};

  // getter 返回精确类型，setter 接受宽泛类型
  get timeout(): number {
    return (this._data.timeout as number) ?? 5000;
  }

  set timeout(value: number | string) {
    this._data.timeout = typeof value === "string" ? parseInt(value, 10) : value;
  }

  get debug(): boolean {
    return (this._data.debug as boolean) ?? false;
  }

  set debug(value: boolean | string) {
    this._data.debug = value === true || value === "true";
  }
}
```

### 2.6 与 JavaScript 的对比

```javascript
// JavaScript：getter/setter 本来就可以有不同类型
class Box {
  #value = "";
  get value() { return this.#value; }
  set value(v) { this.#value = String(v); }
}

// TypeScript 5.1 之前：强制类型相同（限制过于严格）
// TypeScript 5.1+：与 JavaScript 行为一致
```

## 三、注意事项与常见陷阱

1. **getter/setter 类型可以不同**：但要保证语义一致（setter 的类型应该是 getter 类型的超集）
2. **尾部元组元素简化了可变元组类型**：实际使用中感知不大
3. **升级前检查 getter/setter 的类型兼容性**：旧代码可能依赖"必须相同"的约束
4. **`Object.keys` 仍然是 `string[]`**：这是 TypeScript 的设计决策，不是 bug
5. **JSX 配置**：`jsx: "react-jsx"` 配合 `jsxImportSource` 不需要 `import React`
