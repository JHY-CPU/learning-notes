# this 参数类型

## 一、概念说明

TypeScript 允许在函数参数中**显式声明 `this` 的类型**，以确保函数在正确的上下文中调用。`this` 参数必须是函数的第一个参数，且在编译后会被移除。

## 二、具体用法

### 2.1 显式 this 参数

```typescript
interface User {
  name: string;
  greet(this: User): string;
}

const user: User = {
  name: "Alice",
  greet(this: User) {
    return `Hello, I'm ${this.name}`;
  },
};

console.log(user.greet());
// 输出: Hello, I'm Alice
```

### 2.2 回调中的 this

```typescript
class Button {
  label: string;
  constructor(label: string) {
    this.label = label;
  }

  // 显式声明 this 类型
  click(this: Button): void {
    console.log(`${this.label} 被点击`);
  }
}

const btn = new Button("提交");
btn.click();
// 输出: 提交 被点击

// 如果 this 类型不匹配，编译报错
// const click = btn.click;
// click(); // ❌ this 不是 Button 类型
```

### 2.3 箭头函数的 this

```typescript
class Counter {
  count = 0;

  // 箭头函数自动绑定 this，不需要显式声明
  increment = (): void => {
    this.count++;
  };

  // 普通方法需要显式 this
  decrement(this: Counter): void {
    this.count--;
  }
}

const counter = new Counter();
counter.increment();
counter.increment();
console.log(counter.count);
// 输出: 2
```

## 三、注意事项与常见陷阱

1. **`this` 参数是假参数**：不计入函数长度，运行时不存在
2. **箭头函数不需要 `this` 参数**：自动绑定定义时的 `this`
3. **回调中的 `this` 丢失**：方法作为回调传递时 `this` 可能丢失
4. **`--noImplicitThis`**：启用后隐式 `this` 会报错，必须显式声明
