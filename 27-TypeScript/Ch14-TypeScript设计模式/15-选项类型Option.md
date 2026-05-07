# 选项类型Option

## 一、概念说明

Option 类型表示可能存在或不存在的值，替代 `null` 和 `undefined`。TypeScript 中可以用 Discriminated Union 实现类型安全的 Option。

## 二、具体用法

### 2.1 基本 Option 类型

```typescript
type Option<T> =
  | { some: true; value: T }
  | { some: false };

function some<T>(value: T): Option<T> {
  return { some: true, value };
}

function none<T>(): Option<T> {
  return { some: false };
}
```

### 2.2 使用示例

```typescript
function findUser(id: number): Option<User> {
  const user = db.find(u => u.id === id);
  return user ? some(user) : none();
}

// 消费 Option
const result = findUser(1);

if (result.some) {
  console.log(result.value.name); // User
} else {
  console.log('用户不存在');
}
```

### 2.3 Option 操作

```typescript
function map<T, U>(opt: Option<T>, fn: (value: T) => U): Option<U> {
  if (!opt.some) return none();
  return some(fn(opt.value));
}

function flatMap<T, U>(opt: Option<T>, fn: (value: T) => Option<U>): Option<U> {
  if (!opt.some) return none();
  return fn(opt.value);
}

function getOrElse<T>(opt: Option<T>, defaultValue: T): T {
  return opt.some ? opt.value : defaultValue;
}

function unwrap<T>(opt: Option<T>): T {
  if (!opt.some) throw new Error('Option 是 None');
  return opt.value;
}

// 链式调用
const name = map(findUser(1), u => u.name);
const greeting = getOrElse(map(name, n => `Hello, ${n}`), 'Hello, 陌生人');
```

### 2.4 替代 null 检查

```typescript
// 不安全
function getUser(id: number): User | null { return null; }
const user = getUser(1);
// user.name; // 运行时错误

// 安全
function getUserSafe(id: number): Option<User> { return none(); }
const result = getUserSafe(1);
if (result.some) {
  result.value.name; // 类型安全
}
```

## 三、注意事项与常见陷阱

1. **Option 替代 `null`**：更安全的空值处理
2. **`some` 和 `none` 是构造函数**
3. **模式匹配消费 Option**
4. **`getOrElse` 提供默认值**
5. **不要在 Option 中放 `undefined`**
