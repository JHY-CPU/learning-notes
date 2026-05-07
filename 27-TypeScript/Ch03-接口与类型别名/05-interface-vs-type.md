# interface vs type 区别对比

## 一、概念说明

`interface` 和 `type` 都可以定义对象类型，但有重要区别。理解这些区别有助于在正确场景选择正确的工具。

## 二、具体用法

### 2.1 核心区别对比表

```typescript
// interface 能做的
interface User {
  name: string;
}
interface User { age: number; } // ✅ 声明合并
interface Admin extends User { role: string; } // ✅ 继承
class AdminImpl implements Admin { // ✅ implements
  name = "admin";
  age = 30;
  role = "super";
}

// type 能做的（interface 不能）
type ID = string | number;           // ✅ 联合类型
type Pair = [string, number];        // ✅ 元组类型
type Fn = (x: number) => string;     // ✅ 函数类型
type Keys = "a" | "b" | "c";        // ✅ 字面量联合
type Mapped = { [K in Keys]: number }; // ✅ 映射类型
```

### 2.2 选择策略

```typescript
// ✅ 场景 1：定义对象形状 → 用 interface
interface ApiResponse {
  data: unknown;
  status: number;
  message: string;
}

// ✅ 场景 2：联合类型、元组 → 用 type
type Status = "pending" | "success" | "error";
type Coord = [number, number, number];

// ✅ 场景 3：工具函数的参数类型 → 用 type
type EventHandler<T> = (event: T) => void;

// ✅ 场景 4：需要声明合并 → 用 interface
interface Window {
  myGlobalVar: string;
}

console.log("选择原则: 对象用 interface, 其他用 type");
// 输出: 选择原则: 对象用 interface, 其他用 type
```

### 2.3 性能差异

```typescript
// 大型联合类型：type 比 interface 快
// 因为 interface 的交叉类型需要计算结构合并

// 小型对象：无明显差异
// 优先考虑可读性和一致性
```

## 三、注意事项与常见陷阱

1. **默认用 `interface`**：对象类型首选 `interface`，更符合 OOP 习惯
2. **需要联合/元组时用 `type`**：这些只能用 `type` 表达
3. **团队统一**：一个项目中风格应保持一致
4. **可以混用**：`interface` 可以继承 `type`，反之亦然
