# this 参数类型

## 一、概念说明

TypeScript 允许在函数参数中**显式声明 `this` 的类型**，以确保函数在正确的上下文中调用。`this` 参数必须是函数的第一个参数，且在编译后会被移除。这是 TypeScript 解决 JavaScript 中 `this` 绑定问题的重要机制。

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

### 2.4 回调函数中的 this 类型

```typescript
interface DOMElement {
  addEventListener(
    type: string,
    handler: (this: DOMElement, ev: Event) => void
  ): void;
}

// 确保回调中的 this 指向正确的 DOM 元素
function setupClick(element: DOMElement): void {
  element.addEventListener("click", function (this: DOMElement, ev) {
    console.log(this); // this 指向 element
  });
}
```

### 2.5 --noImplicitThis 编译选项

```typescript
// tsconfig.json 开启 strict 或 noImplicitThis 后

class Timer {
  seconds = 0;

  // ❌ 隐式 this 报错
  // start() {
  //   setInterval(function() {
  //     this.seconds++; // 隐式 any 类型
  //   }, 1000);
  // }

  // ✅ 解决方案 1: 箭头函数
  start() {
    setInterval(() => {
      this.seconds++; // this 正确指向 Timer
    }, 1000);
  }

  // ✅ 解决方案 2: 显式 this 参数
  tick(this: Timer) {
    this.seconds++;
  }
}
```

## 三、注意事项与常见陷阱

1. **`this` 参数是假参数**：不计入函数长度，运行时不存在，编译后被移除
2. **箭头函数不需要 `this` 参数**：自动绑定定义时的 `this`，是处理 `this` 问题最简单的方式
3. **回调中的 `this` 丢失**：方法作为回调传递时 `this` 可能丢失，用 `.bind()` 或箭头函数解决
4. **`--noImplicitThis`**：启用后隐式 `this` 会报错，应在所有项目中开启
5. **`this` 不能用于箭头函数**：箭头函数声明 `this` 参数无意义
6. **Vue/React 中的 `this`**：Vue 选项式 API 大量依赖 `this`，React 函数组件则不需要
