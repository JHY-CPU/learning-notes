# 类型别名 type 详解

## 一、概念说明

类型别名（Type Alias）通过 `type` 关键字为任何类型创建新名称。它可以代表原始类型、对象类型、联合类型、交叉类型、元组类型、函数类型等，是 TypeScript 中最灵活的类型定义方式。

## 二、具体用法

### 2.1 基本类型别名

```typescript
type ID = string | number;
type Username = string;
type Age = number;

type User = {
  id: ID;
  name: Username;
  age: Age;
};

const user: User = { id: "u_001", name: "Alice", age: 25 };
console.log(user);
// 输出: { id: "u_001", name: "Alice", age: 25 }
```

### 2.2 联合类型别名

```typescript
type Result<T> =
  | { success: true; data: T }
  | { success: false; error: string };

function fetchUser(): Result<User> {
  return { success: true, data: { id: 1, name: "Bob", age: 30 } };
}

const result = fetchUser();
if (result.success) {
  console.log(result.data.name);
  // 输出: Bob
}
```

### 2.3 函数类型别名

```typescript
type AsyncHandler = (req: Request) => Promise<Response>;
type Middleware = (next: AsyncHandler) => AsyncHandler;

const logging: Middleware = (next) => async (req) => {
  console.log(`请求: ${req.url}`);
  const res = await next(req);
  return res;
};
```

### 2.4 元组类型别名

```typescript
type Coordinate = [number, number];
type Range = [start: number, end: number];

const point: Coordinate = [10, 20];
const range: Range = [0, 100];

console.log(point, range);
// 输出: [10, 20] [0, 100]
```

## 三、注意事项与常见陷阱

1. **`type` 不能声明合并**：同名 `type` 会报重复定义错误
2. **不能 `implements`**：类不能实现类型别名，只能实现接口
3. **更灵活**：联合类型、元组类型只能用 `type` 定义
4. **调试中显示别名**：IDE 中显示的是别名名称，便于理解
