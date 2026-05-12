# 混合模式 Mixin

## 一、概念说明

Mixin 是一种代码复用模式，通过将多个类的功能"混合"到一个类中，实现类似多重继承的效果。TypeScript 中使用函数返回类的方式来实现 Mixin。Mixin 在需要组合多个不相关功能时比单继承更灵活。

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

### 2.3 实际应用：组合功能

```typescript
type Constructor2<T = {}> = new (...args: any[]) => T;

// 带 ID 的 Mixin
function WithId<TBase extends Constructor>(Base: TBase) {
  return class extends Base {
    readonly id = crypto.randomUUID?.() ?? Math.random().toString(36);
  };
}

// 带日志的 Mixin
function WithLogger<TBase extends Constructor>(Base: TBase) {
  return class extends Base {
    log(message: string) {
      console.log(`[${this.constructor.name}] ${message}`);
    }
  };
}

// 带验证的 Mixin
function WithValidation<TBase extends Constructor>(Base: TBase) {
  return class extends Base {
    validate(): boolean {
      return true;
    }
  };
}

// 组合使用
class Document extends WithLogger(WithId(WithValidation(Base))) {
  title = "";
}

const doc = new Document();
doc.title = "TypeScript 教程";
doc.log(`文档已创建: ${doc.title}`);
// 输出: [Document] 文档已创建: TypeScript 教程
```

### 2.4 Mixin 接口声明

```typescript
// 声明混合后的完整类型
interface UserMixin extends Activatable, Timestamped, Base {
  username: string;
}

// 使用声明
function processUser(user: UserMixin) {
  console.log(user.username, user.isActive, user.createdAt);
}
```

## 三、注意事项与常见陷阱

1. **Mixin 顺序很重要**：后面的 Mixin 会覆盖前面的同名成员
2. **类型推断复杂**：嵌套 Mixin 的类型可能难以推断，需要显式声明接口
3. **运行时开销**：每个 Mixin 都会创建新的类层级
4. **替代方案**：简单场景优先考虑组合（Composition）而非 Mixin
5. **构造函数参数**：Mixin 的 `...args: any[]` 丢失了参数类型信息
6. **调试困难**：深层 Mixin 链在调试器中可能难以追踪
