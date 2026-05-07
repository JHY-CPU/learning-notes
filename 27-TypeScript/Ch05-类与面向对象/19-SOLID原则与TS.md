# SOLID 原则与 TS

## 一、概念说明

SOLID 是面向对象设计的五大原则。TypeScript 的类型系统（接口、泛型、抽象类）为实现这些原则提供了良好支持。遵循 SOLID 原则可以编写更灵活、可维护的代码。

## 二、具体用法

### 2.1 S - 单一职责

```typescript
// 每个类只做一件事
class UserValidator {
  validate(user: { email: string }): boolean {
    return user.email.includes("@");
  }
}

class UserRepository {
  save(user: { email: string }): void {
    console.log(`保存用户: ${user.email}`);
  }
}

const validator = new UserValidator();
const repo = new UserRepository();
const user = { email: "a@b.com" };
if (validator.validate(user)) {
  repo.save(user);
}
// 输出: 保存用户: a@b.com
```

### 2.2 O - 开闭原则

```typescript
// 对扩展开放，对修改关闭
interface Shape {
  area(): number;
}

class Circle implements Shape {
  constructor(public radius: number) {}
  area(): number { return Math.PI * this.radius ** 2; }
}

class Rectangle implements Shape {
  constructor(public w: number, public h: number) {}
  area(): number { return this.w * this.h; }
}

// 新增三角形不需要修改现有代码
class Triangle implements Shape {
  constructor(public base: number, public height: number) {}
  area(): number { return (this.base * this.height) / 2; }
}

function totalArea(shapes: Shape[]): number {
  return shapes.reduce((sum, s) => sum + s.area(), 0);
}

const shapes: Shape[] = [new Circle(5), new Rectangle(4, 6), new Triangle(3, 8)];
console.log(totalArea(shapes).toFixed(2));
// 输出: 114.54
```

### 2.3 L - 里氏替换 / I - 接口隔离 / D - 依赖倒置

```typescript
// 依赖倒置：依赖抽象而非具体实现
interface Logger {
  log(message: string): void;
}

class ConsoleLogger implements Logger {
  log(message: string): void { console.log(`[LOG] ${message}`); }
}

class App {
  constructor(private logger: Logger) {}
  run(): void {
    this.logger.log("应用启动");
  }
}

// 可以注入不同的 Logger 实现
const app = new App(new ConsoleLogger());
app.run();
// 输出: [LOG] 应用启动
```

## 三、注意事项与常见陷阱

1. **不要过度设计**：简单代码不需要复杂模式
2. **接口是依赖倒置的关键**：依赖接口而非具体类
3. **泛型增加灵活性**：但也会增加复杂度
4. **TypeScript 只辅助设计**：SOLID 原则本质上是设计思想
