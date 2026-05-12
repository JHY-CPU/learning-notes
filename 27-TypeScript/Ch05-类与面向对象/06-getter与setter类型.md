# getter 与 setter 类型

## 一、概念说明

TypeScript 支持 `get` 和 `set` 访问器，用于控制属性的读写行为。getter 和 setter 可以有独立的类型约束，实现计算属性、验证逻辑等功能。访问器让类的属性访问更像普通属性，同时在内部执行逻辑。

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

### 2.3 惰性计算

```typescript
class DataLoader {
  private _data: unknown[] | null = null;

  get data(): unknown[] {
    if (this._data === null) {
      console.log("首次访问，加载数据...");
      this._data = [1, 2, 3, 4, 5]; // 模拟加载
    }
    return this._data;
  }
}

const loader = new DataLoader();
console.log(loader.data); // 输出: 首次访问，加载数据... [1, 2, 3, 4, 5]
console.log(loader.data); // 输出: [1, 2, 3, 4, 5]（直接返回缓存）
```

### 2.4 接口约束 getter/setter

```typescript
interface IPoint {
  x: number;
  y: number;
  readonly distance: number; // 只读 getter
}

class Point implements IPoint {
  constructor(public x: number, public y: number) {}

  get distance(): number {
    return Math.sqrt(this.x ** 2 + this.y ** 2);
  }
}

const p = new Point(3, 4);
console.log(p.distance); // 输出: 5
// p.distance = 10; // ❌ 编译错误: 只有 getter
```

## 三、注意事项与常见陷阱

1. **getter 不能有参数**：它是属性访问，不是方法调用
2. **setter 只能有一个参数**：参数类型应与属性类型一致
3. **编译目标**：需要 `target` >= ES5 才支持 getter/setter
4. **`--useDefineForClassFields`**：可能影响 getter/setter 的行为和初始化顺序
5. **getter 应是纯函数**：多次访问 getter 应返回相同结果（除非是惰性计算场景）
6. **性能注意**：频繁访问的 getter 中不应放耗时逻辑
