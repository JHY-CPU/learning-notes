# 私有字段 #field

## 一、概念说明

ES2022 引入的私有字段使用 `#` 前缀声明，提供**真正的运行时私有性**。与 TypeScript 的 `private` 关键字（仅编译时检查）不同，`#` 字段在运行时也无法从外部访问。

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

## 三、注意事项与常见陷阱

1. **真正的私有**：运行时也无法从外部访问，比 `private` 更安全
2. **不能与 `private` 混用**：`private #field` 是非法语法
3. **性能略低**：`#` 字段有运行时开销
4. **编译目标**：需要 ES2022+ 或者 polyfill
