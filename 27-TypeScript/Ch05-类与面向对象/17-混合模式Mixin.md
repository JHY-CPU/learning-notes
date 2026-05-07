# 混合模式 Mixin

## 一、概念说明

Mixin 是一种代码复用模式，通过将多个类的功能"混合"到一个类中，实现类似多重继承的效果。TypeScript 中使用函数返回类的方式来实现 Mixin。

## 二、具体用法

### 2.1 基本 Mixin

```typescript
// 基类
class Base {
  name = "";
}

// Mixin 工厂函数
type Constructor<T = {}> = new (...args: any[]) => T;

function Timestamped<TBase extends Constructor>(Base: TBase) {
  return class extends Base {
    createdAt = new Date();
    updatedAt = new Date();
  };
}

function Activatable<TBase extends Constructor>(Base: TBase) {
  return class extends Base {
    isActive = false;
    activate() { this.isActive = true; }
    deactivate() { this.isActive = false; }
  };
}

// 混合多个 Mixin
class User extends Activatable(Timestamped(Base)) {
  constructor(public username: string) {
    super();
  }
}

const user = new User("alice");
user.activate();
console.log(user.username, user.isActive, user.createdAt);
// 输出: alice true 2024-xx-xxTxx:xx:xx.xxxZ
```

### 2.2 带约束的 Mixin

```typescript
interface Loggable {
  log(message: string): void;
}

function Serializable<TBase extends Constructor<Loggable>>(Base: TBase) {
  return class extends Base {
    serialize(): string {
      const data: string[] = [];
      this.log("序列化中...");
      return JSON.stringify(data);
    }
  };
}
```

## 三、注意事项与常见陷阱

1. **Mixin 顺序很重要**：后面的 Mixin 会覆盖前面的同名成员
2. **类型推断复杂**：嵌套 Mixin 的类型可能难以推断
3. **运行时开销**：每个 Mixin 都会创建新的类
4. **替代方案**：简单场景优先考虑组合而非 Mixin
