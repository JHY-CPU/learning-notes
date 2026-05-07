# 接口 interface 基础

## 一、概念说明

`interface` 用于定义对象的**形状**（Shape），规定对象应包含哪些属性、属性的类型以及是否可选或只读。接口是 TypeScript 中描述对象结构最常用的方式，支持声明合并和继承。

## 二、具体用法

### 2.1 定义接口

```typescript
interface User {
  id: number;
  name: string;
  email: string;
  age: number;
}

const user: User = {
  id: 1,
  name: "张三",
  email: "zhangsan@example.com",
  age: 25,
};

console.log(user);
// 输出: { id: 1, name: "张三", email: "zhangsan@example.com", age: 25 }
```

### 2.2 可选属性

```typescript
interface Config {
  host: string;
  port: number;
  debug?: boolean;       // 可选属性
  timeout?: number;      // 可选属性
}

const config: Config = {
  host: "localhost",
  port: 3000,
};
// debug 和 timeout 可以不提供

console.log(config.debug); // 输出: undefined
```

### 2.3 只读属性

```typescript
interface Point {
  readonly x: number;
  readonly y: number;
}

const point: Point = { x: 10, y: 20 };
// point.x = 5; // ❌ 编译错误: Cannot assign to 'x' because it is readonly

console.log(point);
// 输出: { x: 10, y: 20 }
```

## 三、注意事项与常见陷阱

1. **接口只定义形状**：不包含实现代码，编译后消失
2. **可选属性访问需检查**：访问前检查 `undefined` 或用可选链
3. **`readonly` 只防直接赋值**：不防嵌套对象的修改
4. **接口可被声明合并**：同名接口会自动合并属性
