# Partial 与 Required

## 一、概念说明

`Partial<T>` 将类型 T 的所有属性变为**可选**，`Required<T>` 将所有属性变为**必选**。它们是 TypeScript 内置的工具类型，基于映射类型实现。

## 二、具体用法

### 2.1 Partial 基本用法

```typescript
interface User {
  id: number;
  name: string;
  email: string;
}

// Partial<User> = { id?: number; name?: string; email?: string; }
type PartialUser = Partial<User>;

// 创建部分用户数据
const userData: PartialUser = {
  name: "Alice",
};
// id 和 email 可以省略

// 典型应用：更新函数
function updateUser(id: number, updates: Partial<User>): void {
  console.log(`更新用户 ${id}: ${JSON.stringify(updates)}`);
}

updateUser(1, { name: "Bob" });
// 输出: 更新用户 1: {"name":"Bob"}
updateUser(2, { email: "new@email.com" });
// 输出: 更新用户 2: {"email":"new@email.com"}
```

### 2.2 Required 基本用法

```typescript
interface Config {
  host?: string;
  port?: number;
  debug?: boolean;
}

// Required<Config> = { host: string; port: number; debug: boolean; }
type FullConfig = Required<Config>;

const config: FullConfig = {
  host: "localhost",
  port: 3000,
  debug: true,
};
// 所有属性必须提供

console.log(config);
// 输出: { host: "localhost", port: 3000, debug: true }
```

### 2.3 Partial 实现原理

```typescript
// Partial<T> 的源码定义
type MyPartial<T> = {
  [P in keyof T]?: T[P];
};

// Required<T> 的源码定义
type MyRequired<T> = {
  [P in keyof T]-?: T[P];
};

// -? 表示移除可选修饰符
```

## 三、注意事项与常见陷阱

1. **Partial 是浅层的**：不会递归地使嵌套对象属性可选
2. **深 Partial 需自定义**：需要递归工具类型实现
3. **Required 移除可选**：`-?` 语法移除 `?` 修饰符
4. **Partial 配合 Pick**：`Partial<Pick<T, K>>` 只让部分属性可选
