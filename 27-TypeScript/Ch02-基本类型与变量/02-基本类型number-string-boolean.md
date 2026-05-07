# 基本类型 number string boolean

## 一、概念说明

TypeScript 的三种最基本类型对应 JavaScript 的三种原始类型：`number`（数字）、`string`（字符串）、`boolean`（布尔值）。类型注解使用冒号语法 `变量名: 类型`。

## 二、具体用法

### 2.1 number 类型

```typescript
// number 包含整数、浮点数、Infinity、NaN
let age: number = 25;
let price: number = 99.99;
let hex: number = 0xff;        // 十六进制
let binary: number = 0b1010;   // 二进制
let octal: number = 0o744;     // 八进制
let big: bigint = 100n;        // 大整数（单独类型）

console.log(age, price);
// 输出: 25 99.99
console.log(hex, binary, octal);
// 输出: 255 10 484
```

### 2.2 string 类型

```typescript
let greeting: string = "Hello";
let name: string = 'TypeScript';
let template: string = `${greeting}, ${name}!`;

// 模板字符串多行
let multiLine: string = `
  第一行
  第二行
  ${1 + 1}
`;

console.log(template);
// 输出: Hello, TypeScript!
console.log(multiLine.trim());
// 输出: 第一行\n第二行\n2
```

### 2.3 boolean 类型

```typescript
let isActive: boolean = true;
let isDeleted: boolean = false;

// 布尔运算
let a = true;
let b = false;
console.log(a && b);  // 输出: false
console.log(a || b);  // 输出: true
console.log(!a);      // 输出: false
```

## 三、注意事项与常见陷阱

1. **`number` 包括浮点数**：TypeScript 没有 `int`/`float` 区分
2. **`bigint` 是独立类型**：`100n` 是 `bigint`，不兼容 `number`
3. **字符串不可变**：所有字符串方法返回新字符串
4. **布尔值只有 `true`/`false`**：`Boolean(0)` 是运行时行为，类型仍是 `boolean`
