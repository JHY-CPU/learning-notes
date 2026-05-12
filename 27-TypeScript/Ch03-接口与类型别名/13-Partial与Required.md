# Partial 与 Required

## 一、概念说明

`Partial<T>` 将类型 T 的所有属性变为**可选**，`Required<T>` 将所有属性变为**必选**。它们是 TypeScript 内置的工具类型，基于映射类型实现。`Partial` 常用于更新操作，`Required` 用于确保完整配置。

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

### 2.4 深层 Partial

```typescript
// Partial 是浅层的，深层需要递归实现
type DeepPartial<T> = T extends object ? {
  [P in keyof T]?: DeepPartial<T[P]>;
} : T;

interface NestedConfig {
  database: {
    host: string;
    port: number;
    credentials: {
      username: string;
      password: string;
    };
  };
  cache: {
    enabled: boolean;
    ttl: number;
  };
}

// 只更新部分嵌套配置
const partialConfig: DeepPartial<NestedConfig> = {
  database: {
    host: "new-host",
    // 其他字段可省略
  },
};
```

### 2.5 Partial 与 Pick 结合

```typescript
// 只让部分属性可选
type PartialPick<T, K extends keyof T> = Omit<T, K> & Partial<Pick<T, K>>;

interface CreateUser {
  name: string;
  email: string;
  age: number;
  phone: string;
}

// 创建用户时 phone 可选
type CreateUserInput = PartialPick<CreateUser, "phone" | "age">;

const input: CreateUserInput = {
  name: "Alice",
  email: "alice@example.com",
  // phone 和 age 可省略
};
```

## 三、注意事项与常见陷阱

1. **Partial 是浅层的**：不会递归地使嵌套对象属性可选，深层需自定义 `DeepPartial`
2. **深 Partial 需自定义**：需要递归工具类型实现，但要注意性能和类型深度限制
3. **Required 移除可选**：`-?` 语法移除 `?` 修饰符，使所有属性变为必选
4. **Partial 配合 Pick**：`Partial<Pick<T, K>>` 只让部分属性可选，更灵活
5. **Partial 与默认值**：Partial 常配合默认值使用，确保最终对象完整
6. **性能考虑**：过度嵌套的深层 Partial 可能影响类型检查性能
