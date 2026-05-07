# 类型体操实例 - 类型安全 EventEmitter

## 一、概念说明

实现一个类型安全的事件发射器，确保事件名称和回调参数类型在编译时匹配。使用泛型约束事件映射表。

## 二、具体用法

### 2.1 事件映射接口

```typescript
interface EventMap {
  click: { x: number; y: number };
  focus: undefined;
  message: { text: string; sender: string };
}
```

### 2.2 类型安全实现

```typescript
class TypedEmitter<T extends Record<string, any>> {
  private listeners = new Map<string, Set<Function>>();

  on<K extends keyof T>(event: K, handler: (data: T[K]) => void): void {
    if (!this.listeners.has(event as string)) {
      this.listeners.set(event as string, new Set());
    }
    this.listeners.get(event as string)!.add(handler);
  }

  emit<K extends keyof T>(event: K, data: T[K]): void {
    this.listeners.get(event as string)?.forEach(fn => fn(data));
  }
}

const emitter = new TypedEmitter<EventMap>();

emitter.on("click", (data) => {
  console.log(`点击位置: (${data.x}, ${data.y})`);
});

emitter.emit("click", { x: 10, y: 20 });
// 输出: 点击位置: (10, 20)

// emitter.emit("click", { x: "a" }); // ❌ 类型错误
// emitter.emit("unknown", {});        // ❌ 未知事件
```

## 三、注意事项与常见陷阱

1. **事件映射表**：定义所有事件及其数据类型
2. **泛型约束确保类型安全**：回调参数类型与事件数据类型匹配
3. **`undefined` 事件**：无数据事件用 `undefined`
4. **运行时无类型检查**：仍需运行时验证（如处理外部输入）
