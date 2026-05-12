# 类型体操实例 - 类型安全 EventEmitter

## 一、概念说明

实现一个**类型安全**的事件发射器，确保事件名称、回调参数类型和事件数据类型在编译时完全匹配。使用泛型约束事件映射表（Event Map），避免运行时的类型错误。这是 Node.js `EventEmitter` 浏览器 `addEventListener` 等事件系统的类型安全版本。

## 二、具体用法

### 2.1 事件映射接口定义

```typescript
// 定义事件及其数据类型
interface AppEvents {
  login: { userId: string; timestamp: number };
  logout: { userId: string };
  error: { code: number; message: string };
  click: { x: number; y: number };
  resize: { width: number; height: number };
  ready: undefined; // 无数据事件
}
```

### 2.2 类型安全 EventEmitter 实现

```typescript
type EventMap = Record<string, any>;

class TypedEmitter<T extends EventMap> {
  private listeners = new Map<keyof T, Set<(data: any) => void>>();

  // 注册事件处理器，参数类型自动推断
  on<K extends keyof T>(event: K, handler: (data: T[K]) => void): this {
    if (!this.listeners.has(event)) {
      this.listeners.set(event, new Set());
    }
    this.listeners.get(event)!.add(handler);
    return this;
  }

  // 移除事件处理器
  off<K extends keyof T>(event: K, handler: (data: T[K]) => void): this {
    this.listeners.get(event)?.delete(handler);
    return this;
  }

  // 触发事件，数据类型必须匹配
  emit<K extends keyof T>(event: K, data: T[K]): void {
    this.listeners.get(event)?.forEach(fn => fn(data));
  }

  // 一次性监听
  once<K extends keyof T>(event: K, handler: (data: T[K]) => void): this {
    const wrapper = (data: T[K]) => {
      this.off(event, wrapper);
      handler(data);
    };
    return this.on(event, wrapper);
  }
}
```

### 2.3 使用示例

```typescript
const emitter = new TypedEmitter<AppEvents>();

// 类型安全的事件监听
emitter.on("login", (data) => {
  console.log(`用户 ${data.userId} 登录，时间: ${data.timestamp}`);
});

emitter.on("click", (data) => {
  console.log(`点击位置: (${data.x}, ${data.y})`);
});

emitter.on("ready", (data) => {
  // data 类型为 undefined
  console.log("应用就绪");
});

// 类型安全的事件触发
emitter.emit("login", { userId: "u123", timestamp: Date.now() });
emitter.emit("click", { x: 100, y: 200 });
emitter.emit("ready", undefined);

// 以下全部是编译错误
// emitter.emit("login", { userId: 123 });        // 错误：userId 应为 string
// emitter.emit("click", { x: "a", y: 100 });     // 错误：x 应为 number
// emitter.emit("unknown", {});                    // 错误：unknown 不在事件映射中
// emitter.emit("login", { timestamp: 123 });      // 错误：缺少 userId
```

### 2.4 通配符事件监听

```typescript
// 监听所有事件
class WildcardEmitter<T extends EventMap> extends TypedEmitter<T> {
  private wildcardHandlers = new Set<(event: keyof T, data: T[keyof T]) => void>();

  onAny(handler: (event: keyof T, data: T[keyof T]) => void): this {
    this.wildcardHandlers.add(handler);
    return this;
  }

  emit<K extends keyof T>(event: K, data: T[K]): void {
    super.emit(event, data);
    this.wildcardHandlers.forEach(fn => fn(event, data));
  }
}
```

### 2.5 实际应用：前端组件通信

```typescript
// 表单组件事件
interface FormEvents {
  submit: { values: Record<string, unknown>; timestamp: number };
  reset: undefined;
  fieldChange: { field: string; value: unknown; oldValue: unknown };
  validationError: { field: string; error: string };
}

class TypedForm {
  private emitter = new TypedEmitter<FormEvents>();

  on<K extends keyof FormEvents>(event: K, handler: (data: FormEvents[K]) => void) {
    this.emitter.on(event, handler);
  }

  submit(values: Record<string, unknown>) {
    this.emitter.emit("submit", { values, timestamp: Date.now() });
  }

  // 编译时保证事件数据正确
  // this.emitter.emit("submit", { values: {} }); // 编译错误：缺少 timestamp
}
```

### 2.6 与 JavaScript 的对比

```javascript
// JavaScript EventEmitter：无类型保护
const emitter = new EventEmitter();
emitter.on("login", (data) => {
  console.log(data.userId); // 运行时才知道 data 有没有 userId
});
emitter.emit("login", { wrong: "data" }); // 无编译错误，运行时出错

// TypeScript TypedEmitter：编译时保证类型安全
// const emitter = new TypedEmitter<AppEvents>();
// emitter.emit("login", { wrong: "data" }); // 编译错误
```

## 三、注意事项与常见陷阱

1. **事件映射表是类型基础**：必须明确定义所有事件及对应数据类型，`undefined` 表示无数据事件
2. **泛型约束确保类型安全**：`K extends keyof T` 确保事件名合法，`T[K]` 确保数据类型匹配
3. **无数据事件用 `undefined`**：不要用 `{}` 或 `void`，`undefined` 更准确
4. **继承和组合**：可以继承 `TypedEmitter` 添加通配符等功能
5. **运行时仍需验证**：外部数据（如 WebSocket 消息）入站时需运行时校验
6. **内存管理**：及时 `off` 移除事件监听器，避免内存泄漏
