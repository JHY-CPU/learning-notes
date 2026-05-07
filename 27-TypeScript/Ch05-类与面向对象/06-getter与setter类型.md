# getter 与 setter 类型

## 一、概念说明

TypeScript 支持 `get` 和 `set` 访问器，用于控制属性的读写行为。getter 和 setter 可以有独立的类型约束，实现计算属性、验证逻辑等功能。

## 二、具体用法

### 2.1 基本 getter/setter

```typescript
class Temperature {
  private _celsius: number;

  constructor(celsius: number) {
    this._celsius = celsius;
  }

  // getter：只读访问
  get fahrenheit(): number {
    return this._celsius * 9 / 5 + 32;
  }

  // setter：带验证
  set celsius(value: number) {
    if (value < -273.15) {
      throw new Error("温度不能低于绝对零度");
    }
    this._celsius = value;
  }

  get celsius(): number {
    return this._celsius;
  }
}

const temp = new Temperature(100);
console.log(`${temp.celsius}°C = ${temp.fahrenheit}°F`);
// 输出: 100°C = 212°F

temp.celsius = 0;
console.log(`${temp.celsius}°C = ${temp.fahrenheit}°F`);
// 输出: 0°C = 32°F
```

### 2.2 计算属性

```typescript
class Rectangle {
  constructor(public width: number, public height: number) {}

  get area(): number {
    return this.width * this.height;
  }

  get perimeter(): number {
    return 2 * (this.width + this.height);
  }
}

const rect = new Rectangle(10, 5);
console.log(`面积: ${rect.area}, 周长: ${rect.perimeter}`);
// 输出: 面积: 50, 周长: 30
```

## 三、注意事项与常见陷阱

1. **getter 不能有参数**：它是属性访问，不是方法调用
2. **setter 只能有一个参数**：参数类型应与属性类型一致
3. **编译目标**：需要 `target` >= ES5 才支持 getter/setter
4. **`--useDefineForClassFields`**：可能影响 getter/setter 的行为
