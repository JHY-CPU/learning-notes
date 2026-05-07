# Pick 与 Omit

## 一、概念说明

`Pick<T, K>` 从类型 T 中**选取**指定的属性 K 创建新类型，`Omit<T, K>` 从类型 T 中**排除**指定的属性 K 创建新类型。两者互补，用于灵活地派生子类型。

## 二、具体用法

### 2.1 Pick 基本用法

```typescript
interface User {
  id: number;
  name: string;
  email: string;
  password: string;
  createdAt: Date;
}

// 只选取 id 和 name
type UserPreview = Pick<User, "id" | "name">;
// { id: number; name: string; }

const preview: UserPreview = { id: 1, name: "Alice" };
console.log(preview);
// 输出: { id: 1, name: "Alice" }

// 应用：API 响应不暴露敏感字段
function getUserPreview(id: number): UserPreview {
  return { id, name: "Alice" };
}
```

### 2.2 Omit 基本用法

```typescript
// 排除 password
type PublicUser = Omit<User, "password">;
// { id: number; name: string; email: string; createdAt: Date; }

const publicUser: PublicUser = {
  id: 1,
  name: "Alice",
  email: "a@b.com",
  createdAt: new Date(),
};

console.log(publicUser.name);
// 输出: Alice
// publicUser.password; // ❌ 编译错误
```

### 2.3 Pick 和 Omit 实现原理

```typescript
// Pick<T, K> 源码
type MyPick<T, K extends keyof T> = {
  [P in K]: T[P];
};

// Omit<T, K> 源码
type MyOmit<T, K extends keyof any> = {
  [P in keyof T as P extends K ? never : P]: T[P];
};

// as + never 实现属性过滤
```

### 2.4 组合使用

```typescript
// 创建可更新的字段类型
type UpdatableUser = Omit<Partial<User>, "id" | "createdAt">;
// { name?: string; email?: string; password?: string; }

function updateUser(id: number, data: UpdatableUser): void {
  console.log(`更新用户 ${id}: ${JSON.stringify(data)}`);
}

updateUser(1, { name: "新名字" });
// 输出: 更新用户 1: {"name":"新名字"}
```

## 三、注意事项与常见陷阱

1. **`K extends keyof T`**：Pick 的 K 必须是 T 的键，Omit 没有此限制
2. **组合 Pick + Partial**：创建可选的属性子集
3. **Omit 保留多余属性**：不报错未知属性，只是不在结果中
4. **性能注意**：大型接口的 Pick/Omit 计算可能有性能影响
