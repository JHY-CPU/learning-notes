# 私有字段 #field

## 一、概念说明

ES2022 引入的私有字段使用 `#` 前缀声明，提供**真正的运行时私有性**。与 TypeScript 的 `private` 关键字（仅编译时检查）不同，`#` 字段在运行时也无法从外部访问。这是 JavaScript 原生的私有性机制。

## 二、具体用法

### 2.1 基本私有字段

```typescript
class Counter {
  #count = 0; // 真正的私有字段

  increment(): void {
    this.#count++;
  }

  getCount(): number {
    return this.#count;
  }
}

const counter = new Counter();
counter.increment();
counter.increment();
console.log(counter.getCount()); // 输出: 2
// counter.#count; // ❌ 编译错误: 私有字段
// counter["#count"]; // ❌ 运行时也无法访问
```

### 2.2 私有方法

```typescript
class Logger {
  #prefix: string;

  constructor(prefix: string) {
    this.#prefix = prefix;
  }

  #format(message: string): string {
    return `[${this.#prefix}] ${message}`;
  }

  log(message: string): void {
    console.log(this.#format(message));
  }
}

const logger = new Logger("APP");
logger.log("启动完成");
// 输出: [APP] 启动完成
```

### 2.3 私有字段与 instanceof

```typescript
class DataStore {
  #data: Map<string, unknown> = new Map();

  set(key: string, value: unknown): void {
    this.#data.set(key, value);
  }

  get(key: string): unknown {
    return this.#data.get(key);
  }
}

const store = new DataStore();
store.set("name", "Alice");
console.log(store.get("name"));
// 输出: Alice
console.log(store instanceof DataStore); // 输出: true
```

### 2.4 与 private 关键字的对比

```typescript
// private —— 编译时检查，运行时可绕过
class WithPrivate {
  private secret = "hidden";
}

const wp = new WithPrivate();
// wp.secret;        // ❌ 编译错误
console.log((wp as any).secret); // ✅ 运行时可访问

// # —— 运行时私有，不可绕过
class WithHash {
  #secret = "hidden";
}

const wh = new WithHash();
// wh.#secret;       // ❌ 编译错误
// (wh as any).#secret; // ❌ 语法错误
// (wh as any)["#secret"]; // ✅ 返回 undefined（不是真正的字段名）
```

### 2.5 私有字段与序列化

```typescript
class User {
  #password: string;

  constructor(public name: string, password: string) {
    this.#password = password;
  }

  checkPassword(input: string): boolean {
    return this.#password === input;
  }
}

const user = new User("Alice", "secret123");
console.log(JSON.stringify(user)); // 输出: {"name":"Alice"}
// #password 不会被序列化
```

## 三、注意事项与常见陷阱

1. **真正的私有**：运行时也无法从外部访问，比 `private` 更安全
2. **不能与 `private` 混用**：`private #field` 是非法语法
3. **性能略低**：`#` 字段有运行时开销，大量实例时需注意
4. **编译目标**：需要 ES2022+ 或者 polyfill
5. **不会被 JSON.stringify 序列化**：`#` 字段天然不会出现在序列化结果中
6. **类中同名 `#` 字段是独立的**：不同类的 `#field` 是不同的私有槽位
