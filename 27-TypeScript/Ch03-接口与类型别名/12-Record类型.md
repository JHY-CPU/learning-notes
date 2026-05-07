# Record 类型

## 一、概念说明

`Record<K, V>` 是 TypeScript 内置的工具类型，用于创建一个键类型为 K、值类型为 V 的对象类型。它比索引签名更类型安全，因为 K 可以是联合类型而不仅限于 `string` 或 `number`。

## 二、具体用法

### 2.1 基本用法

```typescript
// 创建键为 string、值为 number 的对象类型
const scores: Record<string, number> = {
  math: 95,
  english: 88,
  physics: 92,
};

console.log(scores.math);
// 输出: 95
```

### 2.2 联合类型作为键

```typescript
type Role = "admin" | "editor" | "viewer";

// 键必须是 Role 的所有值，值类型为权限数组
const permissions: Record<Role, string[]> = {
  admin: ["read", "write", "delete"],
  editor: ["read", "write"],
  viewer: ["read"],
};

function canWrite(role: Role): boolean {
  return permissions[role].includes("write");
}

console.log(canWrite("admin"));  // 输出: true
console.log(canWrite("viewer")); // 输出: false
```

### 2.3 配合映射类型

```typescript
interface User {
  id: number;
  name: string;
  email: string;
}

// 创建以 id 为键的用户映射
const usersById: Record<number, User> = {
  1: { id: 1, name: "Alice", email: "a@b.com" },
  2: { id: 2, name: "Bob", email: "b@c.com" },
};

console.log(usersById[1].name);
// 输出: Alice
```

### 2.4 Record 实现原理

```typescript
// Record<K, V> 的源码定义
// type Record<K extends keyof any, V> = {
//   [P in K]: V;
// };

// 自定义更严格的 Record
type StrictRecord<K extends string, V> = {
  [P in K]: V;
};

type Status = "active" | "inactive";
const statuses: StrictRecord<Status, boolean> = {
  active: true,
  inactive: false,
};

console.log(statuses);
// 输出: { active: true, inactive: false }
```

## 三、注意事项与常见陷阱

1. **键必须全有**：`Record<"a" | "b", T>` 要求对象必须同时有 `a` 和 `b` 属性
2. **`keyof any`**：`Record<string, V>` 等价于 `{ [key: string]: V }`
3. **优先于索引签名**：`Record<K, V>` 比索引签名更类型安全
4. **可选 Record**：用 `Partial<Record<K, V>>` 允许部分键
