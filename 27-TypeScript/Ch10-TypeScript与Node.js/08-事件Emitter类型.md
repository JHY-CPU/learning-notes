# 事件Emitter类型

## 一、概念说明

Node.js 的 `EventEmitter` 是事件驱动架构的核心。TypeScript 通过泛型可以为 EventEmitter 定义精确的事件名称和事件参数类型，确保事件的发射和监听都有类型安全。

## 二、具体用法

### 2.1 泛型 EventEmitter

```typescript
import { EventEmitter } from 'node:events';

// 定义事件映射接口
interface AppEvents {
  connection: (socket: WebSocket) => void;
  disconnect: (reason: string) => void;
  error: (err: Error) => void;
  data: (payload: { id: number; content: string }) => void;
}

// 泛型 EventEmitter
class TypedEmitter<TEvents extends Record<string, (...args: any[]) => void>> {
  private emitter = new EventEmitter();

  on<K extends keyof TEvents>(event: K, listener: TEvents[K]) {
    this.emitter.on(event as string, listener);
    return this;
  }

  emit<K extends keyof TEvents>(event: K, ...args: Parameters<TEvents[K]>) {
    return this.emitter.emit(event as string, ...args);
  }

  off<K extends keyof TEvents>(event: K, listener: TEvents[K]) {
    this.emitter.off(event as string, listener);
    return this;
  }
}

// 使用
const app = new TypedEmitter<AppEvents>();

app.on('connection', (socket) => {
  // socket 类型自动推断为 WebSocket
  console.log(socket);
});

app.emit('data', { id: 1, content: 'hello' }); // 类型安全
// app.emit('data', 'wrong'); // 编译错误
```

### 2.2 使用 Node.js 原生类型

```typescript
import { EventEmitter } from 'node:events';

// 声明事件方法类型
class Server extends EventEmitter {
  // 使用 declare 关键字声明事件方法
  on(event: 'connection', listener: (socket: any) => void): this;
  on(event: 'close', listener: () => void): this;
  on(event: string, listener: (...args: any[]) => void): this;
  on(event: string, listener: (...args: any[]) => void): this {
    return super.on(event, listener);
  }

  emit(event: 'connection', socket: any): boolean;
  emit(event: 'close'): boolean;
  emit(event: string, ...args: any[]): boolean;
  emit(event: string, ...args: any[]): boolean {
    return super.emit(event, ...args);
  }
}
```

### 2.3 类型安全的事件总线

```typescript
// 全局事件总线
type EventMap = {
  'user:login': (userId: number) => void;
  'user:logout': (userId: number) => void;
  'order:created': (orderId: string, total: number) => void;
  'notification': (message: string, level: 'info' | 'warn' | 'error') => void;
};

class EventBus {
  private emitter = new EventEmitter();

  subscribe<K extends keyof EventMap>(event: K, handler: EventMap[K]) {
    this.emitter.on(event, handler);
    return () => this.emitter.off(event, handler); // 返回取消订阅函数
  }

  publish<K extends keyof EventMap>(event: K, ...args: Parameters<EventMap[K]>) {
    this.emitter.emit(event, ...args);
  }
}

const bus = new EventBus();

// 使用 — 完整类型安全
const unsubscribe = bus.subscribe('user:login', (userId) => {
  console.log(userId); // number
});

bus.publish('order:created', 'ORD-001', 99.9);
// bus.publish('order:created', 123, 'wrong'); // 编译错误
```

### 2.4 once 与异步等待

```typescript
import { once } from 'node:events';

// 等待事件触发
async function waitForConnection(server: Server) {
  const [socket] = await once(server, 'connection');
  return socket;
}

// 超时版本
async function waitForEvent<T>(
  emitter: EventEmitter,
  event: string,
  timeout: number
): Promise<T> {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error('超时')), timeout);

    emitter.once(event, (data: T) => {
      clearTimeout(timer);
      resolve(data);
    });
  });
}
```

## 三、注意事项与常见陷阱

1. **使用泛型约束事件参数**：确保 `emit` 和 `on` 的类型匹配
2. **`this` 返回类型**：链式调用需要 `return this`
3. **`once` 返回数组**：`await once()` 返回 `[...args]` 需要解构
4. **内存泄漏**：确保移除不再需要的监听器
5. **Node.js 22+ 支持 `EventEmitter` 的泛型**：检查最新 API
