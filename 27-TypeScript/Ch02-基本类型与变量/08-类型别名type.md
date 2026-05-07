# 类型别名 type

## 一、概念说明

`type` 关键字用于创建类型别名（Type Alias），给一个类型起新名字。类型别名可以代表任何类型：原始类型、联合类型、交叉类型、对象类型、函数类型等。它是 TypeScript 中最灵活的类型定义方式。

## 二、具体用法

### 2.1 基本类型别名

```typescript
// 为基本类型起别名
type ID = string | number;
type Username = string;

function findUser(id: ID): Username {
  return `用户_${id}`;
}

console.log(findUser(42));      // 输出: 用户_42
console.log(findUser("abc"));   // 输出: 用户_abc
```

### 2.2 对象类型别名

```typescript
type User = {
  id: number;
  name: string;
  email: string;
  role: "admin" | "user" | "guest";
};

const admin: User = {
  id: 1,
  name: "管理员",
  email: "admin@example.com",
  role: "admin",
};

console.log(admin);
// 输出: { id: 1, name: "管理员", email: "admin@example.com", role: "admin" }
```

### 2.3 函数类型别名

```typescript
type EventHandler = (event: string, data: unknown) => void;

const onClick: EventHandler = (event, data) => {
  console.log(`${event}: ${JSON.stringify(data)}`);
};

onClick("click", { x: 10, y: 20 });
// 输出: click: {"x":10,"y":20}
```

### 2.4 泛型类型别名

```typescript
type Result<T> = {
  data: T;
  error: string | null;
  loading: boolean;
};

const userResult: Result<User> = {
  data: { id: 1, name: "Alice", email: "a@b.com", role: "user" },
  error: null,
  loading: false,
};

console.log(userResult.loading); // 输出: false
```

## 三、注意事项与常见陷阱

1. **`type` 不能被 `implements`**：类只能 `implements` 接口，不能 `implements` 类型别名
2. **`type` 不能声明合并**：同名 `type` 会报重复定义错误
3. **更灵活**：`type` 可以定义联合类型、元组类型等，`interface` 不行
4. **与 `interface` 选择**：优先用 `interface` 定义对象，用 `type` 定义其他类型
