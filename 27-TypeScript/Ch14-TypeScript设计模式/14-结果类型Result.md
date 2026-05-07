# 结果类型Result

## 一、概念说明

Result 类型源自 Rust，表示可能成功或失败的操作。TypeScript 中实现 Result 可以避免异常，用类型系统表达错误。

## 二、具体用法

### 2.1 基本 Result 类型

```typescript
type Result<T, E = Error> =
  | { ok: true; value: T }
  | { ok: false; error: E };

// 构造函数
function ok<T>(value: T): Result<T, never> {
  return { ok: true, value };
}

function err<E>(error: E): Result<never, E> {
  return { ok: false, error };
}
```

### 2.2 使用示例

```typescript
function divide(a: number, b: number): Result<number, string> {
  if (b === 0) return err('除数不能为零');
  return ok(a / b);
}

// 消费 Result
const result = divide(10, 0);

if (result.ok) {
  console.log(result.value); // number
} else {
  console.log(result.error); // string
}

// 模式匹配
function unwrap<T, E>(result: Result<T, E>): T {
  if (result.ok) return result.value;
  throw result.error;
}
```

### 2.3 Result 链式操作

```typescript
function map<T, U, E>(result: Result<T, E>, fn: (value: T) => U): Result<U, E> {
  if (!result.ok) return result;
  return ok(fn(result.value));
}

function flatMap<T, U, E>(
  result: Result<T, E>,
  fn: (value: T) => Result<U, E>
): Result<U, E> {
  if (!result.ok) return result;
  return fn(result.value);
}

// 链式调用
const r = ok(10)
  |> map(%, (v) => v * 2)
  |> flatMap(%, (v) => divide(v, 5));
```

### 2.4 异步 Result

```type
type AsyncResult<T, E = Error> = Promise<Result<T, E>>;

async function fetchUser(id: number): AsyncResult<User, string> {
  try {
    const res = await fetch(`/api/users/${id}`);
    if (!res.ok) return err(`HTTP ${res.status}`);
    return ok(await res.json());
  } catch (e) {
    return err(String(e));
  }
}
```

## 三、注意事项与常见陷阱

1. **Result 替代 try-catch**：预期的失败用 Result，意外的错误用异常
2. **`ok` 和 `err` 是构造函数**：简化 Result 创建
3. **Discriminated Union 确保类型安全**
4. **异步操作用 `Promise<Result<T, E>>`**
5. **不要忘记处理错误分支**
