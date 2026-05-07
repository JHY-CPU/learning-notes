# keyof 运算符

## 一、概念说明

`keyof` 返回对象类型的所有键名组成的**联合类型**。它是类型安全的键访问基础，配合泛型约束实现类型安全的属性访问。

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
```

### 2.2 泛型约束

```typescript
function getProperty<T, K extends keyof T>(obj: T, key: K): T[K] {
  return obj[key];
}

const user: User = { id: 1, name: "Alice", email: "a@b.com" };
console.log(getProperty(user, "name"));
// 输出: Alice
```

### 2.3 keyof any

```typescript
type AnyKey = keyof any;
// string | number | symbol（所有可能的键类型）
```

## 三、注意事项与常见陷阱

1. **只返回公共属性**：`private` 和 `protected` 不包括在内
2. **`keyof T & string`**：排除 `number` 和 `symbol` 键
3. **与索引访问结合**：`T[K]` 获取属性类型
4. **映射类型基础**：`[K in keyof T]` 是核心模式
