# 接口与类 implements

## 一、概念说明

类可以通过 `implements` 关键字实现一个或多个接口，承诺提供接口定义的所有属性和方法。这是 TypeScript 实现面向对象设计中**契约模式**的核心机制。

## 二、具体用法

### 2.1 基本实现

```typescript
interface Drawable {
  draw(): void;
  color: string;
}

class Circle implements Drawable {
  color: string;
  radius: number;

  constructor(color: string, radius: number) {
    this.color = color;
    this.radius = radius;
  }

  draw(): void {
    console.log(`画一个 ${this.color} 的圆，半径 ${this.radius}`);
  }
}

const circle = new Circle("红色", 10);
circle.draw();
// 输出: 画一个 红色 的圆，半径 10
```

### 2.2 实现多个接口

```typescript
interface Printable {
  print(): void;
}

interface Serializable {
  serialize(): string;
}

class Document implements Printable, Serializable {
  content: string;

  constructor(content: string) {
    this.content = content;
  }

  print(): void {
    console.log(this.content);
  }

  serialize(): string {
    return JSON.stringify({ content: this.content });
  }
}

const doc = new Document("Hello");
doc.print();
// 输出: Hello
console.log(doc.serialize());
// 输出: {"content":"Hello"}
```

### 2.3 接口约束类行为

```typescript
interface Cache<T> {
  get(key: string): T | undefined;
  set(key: string, value: T): void;
  delete(key: string): boolean;
}

class MemoryCache<T> implements Cache<T> {
  private store = new Map<string, T>();

  get(key: string): T | undefined {
    return this.store.get(key);
  }

  set(key: string, value: T): void {
    this.store.set(key, value);
  }

  delete(key: string): boolean {
    return this.store.delete(key);
  }
}

const cache = new MemoryCache<number>();
cache.set("count", 42);
console.log(cache.get("count"));
// 输出: 42
```

## 三、注意事项与常见陷阱

1. **实现必须包含所有成员**：遗漏任何属性或方法会编译报错
2. **结构兼容即可**：不需要显式 `implements`，只要结构匹配就兼容
3. **接口不能约束构造函数参数**：接口只约束实例成员
4. **`implements` 不影响继承**：类仍可 `extends` 其他类
