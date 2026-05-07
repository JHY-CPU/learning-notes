# enum改进

## 一、概念说明

TypeScript 5.0+ 对枚举进行了多项改进，包括所有枚举变为联合枚举、独立枚举支持等。这些改进使枚举在类型系统中更加一致和有用。

## 二、具体用法

### 2.1 联合枚举

```typescript
// TypeScript 5.0+：所有枚举自动成为联合枚举
enum Direction {
  Up = 0,
  Down = 1,
  Left = 2,
  Right = 3,
}

// Direction 类型是 0 | 1 | 2 | 3 的联合
function move(dir: Direction) {
  // 类型收窄
  if (dir === Direction.Up) {
    // dir 被收窄为 Direction.Up（值为 0）
  }
}

// 字符串枚举也是联合
enum Status {
  Active = 'ACTIVE',
  Inactive = 'INACTIVE',
}

// Status 类型是 'ACTIVE' | 'INACTIVE'
```

### 2.2 独立枚举

```typescript
// TypeScript 5.0 支持枚举的独立使用
// 不需要作为成员的类型注解

enum LogLevel {
  Debug,
  Info,
  Warn,
  Error,
}

// 可以直接用枚举值
function log(level: LogLevel, message: string) {
  if (level >= LogLevel.Warn) {
    console.error(message);
  }
}
```

### 2.3 const enum 改进

```typescript
// const enum 在编译时内联
const enum Direction {
  Up = 'UP',
  Down = 'DOWN',
}

// 编译后：const dir = "UP";
const dir = Direction.Up;

// 注意：isolatedModules 模式下有兼容性问题
```

### 2.4 枚举与联合类型的对比

```typescript
// 枚举 — 有运行时对象
enum Color {
  Red = 'RED',
  Green = 'GREEN',
  Blue = 'BLUE',
}
console.log(Color.Red); // 'RED' — 运行时可用

// 联合类型 — 只有编译时类型
type Color = 'RED' | 'GREEN' | 'BLUE';
// 没有运行时对象

// 推荐：优先使用联合类型
// 除非需要运行时枚举对象
```

### 2.5 枚举类型收窄

```typescript
enum Animal {
  Dog,
  Cat,
  Bird,
}

function describe(animal: Animal): string {
  switch (animal) {
    case Animal.Dog: return '汪汪';
    case Animal.Cat: return '喵喵';
    case Animal.Bird: return '叽叽';
  }
}
```

## 三、注意事项与常见陷阱

1. **联合枚举允许类型收窄**：在 switch 中使用
2. **`const enum` 有 `isolatedModules` 兼容性问题**：可能被 esbuild 不支持
3. **字符串枚举优于数字枚举**：调试时有意义
4. **联合类型替代简单枚举**：更简洁，无运行时开销
5. **枚举值应有明确的值**：避免隐式数字赋值
