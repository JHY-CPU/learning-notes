# 装饰器ECMAScript

## 一、概念说明

TC39 Stage 3 装饰器是 ECMAScript 标准的装饰器实现，TypeScript 5.0+ 原生支持。与旧版实验性装饰器不同，新版装饰器使用 `DecoratorContext` 参数，支持类、方法、属性、getter/setter 和自动访问器的装饰。

## 二、具体用法

### 2.1 方法装饰器

```typescript
function log(
  target: (this: any, ...args: any[]) => any,
  context: ClassMethodDecoratorContext
) {
  const name = String(context.name);

  return function (this: any, ...args: any[]) {
    console.log(`调用 ${name}，参数:`, args);
    const result = target.apply(this, args);
    console.log(`${name} 返回:`, result);
    return result;
  };
}

class Calculator {
  @log
  add(a: number, b: number): number {
    return a + b;
  }
}
```

### 2.2 类装饰器

```typescript
function withTimestamp<T extends new (...args: any[]) => any>(
  target: T,
  context: ClassDecoratorContext
) {
  return class extends target {
    createdAt = new Date();
  };
}

@withTimestamp
class User {
  name: string;
  constructor(name: string) {
    this.name = name;
  }
}
```

### 2.3 自动访问器装饰器

```typescript
function validate(min: number, max: number) {
  return function (
    target: undefined,
    context: ClassFieldDecoratorContext
  ) {
    return function (this: any, value: number) {
      if (value < min || value > max) {
        throw new Error(`值必须在 ${min}-${max} 之间`);
      }
      return value;
    };
  };
}

class Age {
  @validate(0, 150)
  accessor value: number = 0;
}
```

### 2.4 装饰器元数据

```typescript
function meta(key: string, value: unknown) {
  return function (target: any, context: DecoratorContext) {
    context.metadata[key] = value;
    return target;
  };
}

@meta('version', '1.0')
class Service {
  @meta('description', '获取用户')
  getUser() {}
}
```

### 2.5 装饰器组合

```typescript
function bound(
  target: (this: any, ...args: any[]) => any,
  context: ClassMethodDecoratorContext
) {
  context.addInitializer(function (this: any) {
    this[context.name] = this[context.name].bind(this);
  });
  return target;
}

function retry(times: number) {
  return function (
    target: (this: any, ...args: any[]) => Promise<any>,
    context: ClassMethodDecoratorContext
  ) {
    return async function (this: any, ...args: any[]) {
      for (let i = 0; i < times; i++) {
        try {
          return await target.apply(this, args);
        } catch (err) {
          if (i === times - 1) throw err;
        }
      }
    };
  };
}

class ApiClient {
  @retry(3)
  @bound
  async fetchData() {
    // 自动重试 3 次
  }
}
```

## 三、注意事项与常见陷阱

1. **移除 `experimentalDecorators: true`**：Stage 3 装饰器不需要
2. **装饰器函数签名不同**：第一个参数不是 `prototype`，第二个参数是 `context`
3. **装饰器执行顺序**：从上到下，从右到左（方法）
4. **`context.metadata` 存储元数据**：可在运行时访问
5. **不要混合新旧装饰器语法**：选择一种统一使用
