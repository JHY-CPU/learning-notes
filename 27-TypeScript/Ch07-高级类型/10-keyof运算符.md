# keyof 运算符

## 一、概念说明

`keyof` 运算符返回对象类型的所有**键名**组成的联合类型。它是类型安全属性访问的基础，配合泛型约束可以实现编译时检查的属性读取和赋值。`keyof` 结合索引访问类型 `T[K]` 是 TypeScript 类型编程最核心的模式之一。

## 二、具体用法

### 2.1 基本用法

```typescript
interface User {
  id: number;
  name: string;
  email: string;
}

type UserKeys = keyof User;
// "id" | "name" | "email"

// keyof 对非对象类型
type NumberKeys = keyof number;     // "toString" | "toFixed" | ...（number 原型方法）
type StringKeys = keyof string;     // "length" | "charAt" | ...
type AnyKeys = keyof any;           // string | number | symbol
```

### 2.2 泛型约束实现类型安全属性访问

```typescript
function getProperty<T, K extends keyof T>(obj: T, key: K): T[K] {
  return obj[key];
}

function setProperty<T, K extends keyof T>(obj: T, key: K, value: T[K]): void {
  obj[key] = value;
}

const user: User = { id: 1, name: "Alice", email: "a@b.com" };

const name = getProperty(user, "name");   // string
setProperty(user, "email", "new@b.com");  // OK
// setProperty(user, "email", 123);       // 错误：number 不能赋给 string
// getProperty(user, "phone");            // 错误："phone" 不在 User 中
```

### 2.3 遍历对象的键和值

```typescript
// 获取所有值的联合类型
type UserValues = User[keyof User];
// number | string

// 类型安全的 pick
function pick<T, K extends keyof T>(obj: T, keys: K[]): Pick<T, K> {
  const result = {} as Pick<T, K>;
  for (const key of keys) {
    result[key] = obj[key];
  }
  return result;
}

const picked = pick(user, ["id", "name"]);
// { id: number; name: string }
```

### 2.4 keyof 与映射类型

```typescript
// Partial 的实现原理
type MyPartial<T> = { [K in keyof T]?: T[K] };

// 重映射键名
type Getters<T> = {
  [K in keyof T as `get${Capitalize<string & K>}`]: () => T[K];
};

// 过滤特定类型的键
type StringKeysOf<T> = {
  [K in keyof T]: T[K] extends string ? K : never;
}[keyof T];

type StringUserKeys = StringKeysOf<User>; // "name" | "email"
```

### 2.5 实际应用：事件处理器映射

```typescript
interface ComponentEvents {
  onClick: (x: number, y: number) => void;
  onFocus: () => void;
  onBlur: () => void;
  onChange: (value: string) => void;
}

// 类型安全的事件绑定
function bindEvents<T extends Record<string, Function>>(
  handlers: T,
  eventName: keyof T
): void {
  // 绑定逻辑
}

bindEvents(
  { onClick: (x: number, y: number) => {}, onFocus: () => {} },
  "onClick"  // OK
);
// bindEvents(handlers, "onScroll"); // 错误：不在事件映射中
```

### 2.6 与 JavaScript 的对比

```javascript
// JavaScript：运行时获取对象的键
const user = { id: 1, name: "Alice", email: "a@b.com" };
Object.keys(user); // ["id", "name", "email"]（字符串数组）
// 注意：Object.keys 返回 string[]，不是联合类型

// TypeScript keyof：编译时获取类型的键联合
// type Keys = keyof User; // "id" | "name" | "email"
// 编译时精确知道所有可能的键，提供自动补全和类型检查
```

## 三、注意事项与常见陷阱

1. **只返回公共属性**：`private` 和 `protected` 成员不包含在 `keyof` 结果中
2. **`keyof any`**：返回 `string | number | symbol`，是最宽泛的键类型
3. **`keyof T & string`**：排除 `number` 和 `symbol` 键，只保留字符串键
4. **索引签名的影响**：`{ [key: string]: number }` 的 `keyof` 是 `string`
5. **`Object.keys` 不是类型安全的**：运行时返回 `string[]`，与 `keyof` 不完全对应
6. **联合类型的 keyof**：`keyof (A | B)` 等于 `keyof A & keyof B`（交集而非并集）
