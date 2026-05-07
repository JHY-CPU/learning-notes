# declare function

## 一、概念说明

`declare function` 声明全局函数的类型签名，不包含实现。用于为 JavaScript 库、全局工具函数等提供类型信息。支持函数重载。

## 二、具体用法

### 2.1 基本函数声明

```typescript
// 全局函数
declare function log(message: string): void;
declare function fetch(url: string, options?: RequestInit): Promise<Response>;
declare function setTimeout(callback: () => void, ms: number): number;

// 使用
log('Hello');           // OK
log(123);               // 错误：参数必须是 string
```

### 2.2 函数重载

```typescript
// 重载 — 同一函数多个签名
declare function createElement(tag: 'a'): HTMLAnchorElement;
declare function createElement(tag: 'div'): HTMLDivElement;
declare function createElement(tag: 'input'): HTMLInputElement;
declare function createElement(tag: string): HTMLElement;

// TypeScript 会根据参数选择正确的重载
const anchor = createElement('a');     // HTMLAnchorElement
const div = createElement('div');      // HTMLDivElement
const unknown = createElement('custom'); // HTMLElement
```

### 2.3 回调函数类型

```typescript
// 使用 type 定义回调类型
type EventCallback = (event: Event) => void;

declare function addEventListener(
  type: string,
  listener: EventCallback,
  options?: boolean | AddEventListenerOptions
): void;

declare function removeEventListener(
  type: string,
  listener: EventCallback
): void;
```

### 2.4 泛型函数

```typescript
// 泛型函数声明
declare function identity<T>(value: T): T;
declare function map<T, U>(array: T[], fn: (item: T) => U): U[];
declare function filter<T>(array: T[], predicate: (item: T) => boolean): T[];

// 使用
const numbers = [1, 2, 3];
const doubled = map(numbers, n => n * 2); // number[]
const strings = map(numbers, n => String(n)); // string[]
```

### 2.5 可变参数

```typescript
declare function concat(...parts: string[]): string;
declare function merge<T>(...objects: Partial<T>[]): T;

// 使用
const greeting = concat('Hello', ' ', 'World');
const config = merge(defaults, overrides, userConfig);
```

## 三、注意事项与常见陷阱

1. **重载顺序很重要**：更具体的签名放在前面
2. **`declare function` 只能声明全局函数**：模块函数用 `declare module`
3. **回调函数的类型要精确定义**：参数和返回值
4. **泛型函数声明确保类型推断**：调用时不需要显式指定类型
5. **可选参数用 `?` 标记**：`param?: type`
