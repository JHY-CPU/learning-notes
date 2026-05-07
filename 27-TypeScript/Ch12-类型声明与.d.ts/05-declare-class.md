# declare class

## 一、概念说明

`declare class` 声明全局类的类型结构，包括构造函数、属性和方法。用于为 JavaScript 中的类提供类型信息，常用于 DOM API、第三方库的类等。

## 二、具体用法

### 2.1 基本类声明

```typescript
// 全局类
declare class EventEmitter {
  // 构造函数
  constructor();

  // 属性
  readonly defaultMaxListeners: number;

  // 方法
  on(event: string, listener: (...args: any[]) => void): this;
  off(event: string, listener: (...args: any[]) => void): this;
  emit(event: string, ...args: any[]): boolean;
  once(event: string, listener: (...args: any[]) => void): this;
  removeAllListeners(event?: string): this;
}
```

### 2.2 带泛型的类

```typescript
declare class Collection<T> {
  constructor(items?: T[]);

  readonly size: number;

  add(item: T): this;
  remove(item: T): boolean;
  has(item: T): boolean;
  forEach(callback: (item: T, index: number) => void): void;
  map<U>(callback: (item: T) => U): Collection<U>;
  filter(predicate: (item: T) => boolean): Collection<T>;
  toArray(): T[];
}
```

### 2.3 抽象类

```typescript
declare abstract class BaseComponent {
  abstract render(): void;

  readonly id: string;
  protected parent: BaseComponent | null;

  mount(container: HTMLElement): void;
  unmount(): void;
  onMount(callback: () => void): void;
}
```

### 2.4 继承

```typescript
declare class Animal {
  constructor(name: string);
  name: string;
  speak(): void;
}

declare class Dog extends Animal {
  breed: string;
  fetch(): void;
}

// 使用
const dog = new Dog('Buddy');
dog.name;    // string
dog.breed;   // string
dog.speak(); // OK
dog.fetch(); // OK
```

### 2.5 静态成员

```typescript
declare class Http {
  static create(baseUrl: string): Http;
  static defaultHeaders: Record<string, string>;

  get<T>(url: string): Promise<T>;
  post<T>(url: string, data: unknown): Promise<T>;
}

// 使用
const http = Http.create('https://api.example.com');
Http.defaultHeaders['Authorization'] = 'Bearer token';
```

### 2.6 实现接口

```typescript
interface Serializable {
  serialize(): string;
  deserialize(data: string): void;
}

declare class DataStore implements Serializable {
  data: Record<string, unknown>;
  serialize(): string;
  deserialize(data: string): void;
  get(key: string): unknown;
  set(key: string, value: unknown): void;
}
```

## 三、注意事项与常见陷阱

1. **`declare class` 不能有实现**：只有类型结构
2. **类可以实现接口**：用 `implements` 关键字
3. **构造函数的参数类型要声明**：`constructor(args: Types)`
4. **`this` 返回类型**：用于链式调用
5. **`readonly` 修饰属性**：不能在外部修改
