# Exclude 与 Extract

## 一、概念说明

`Exclude<T, U>` 从联合类型 T 中**排除**可赋值给 U 的类型成员，`Extract<T, U>` 从联合类型 T 中**提取**可赋值给 U 的类型成员。两者互为补集，用于联合类型的过滤操作。

## 二、具体用法

### 2.1 Exclude 基本用法

```typescript
type AllTypes = string | number | boolean | null | undefined;

// 排除 null 和 undefined
type NonNullableTypes = Exclude<AllTypes, null | undefined>;
// string | number | boolean

const val: NonNullableTypes = "hello"; // ✅
const val2: NonNullableTypes = 42;     // ✅
// const val3: NonNullableTypes = null; // ❌

console.log(val, val2);
// 输出: hello 42
```

### 2.2 Extract 基本用法

```typescript
type StringOrNumber = string | number | boolean;

// 只提取字符串
type OnlyString = Extract<StringOrNumber, string>;
// string

const s: OnlyString = "hello"; // ✅
// const n: OnlyString = 42;   // ❌

console.log(s);
// 输出: hello
```

### 2.3 Exclude 和 Extract 实现原理

```typescript
// Exclude<T, U> 源码
type MyExclude<T, U> = T extends U ? never : T;
// 分布式条件类型：联合类型逐个成员判断

// Extract<T, U> 源码
type MyExtract<T, U> = T extends U ? T : never;

// 示例：Exclude<"a" | "b" | "c", "a">
// = ("a" extends "a" ? never : "a")
// | ("b" extends "a" ? never : "b")
// | ("c" extends "a" ? never : "c")
// = never | "b" | "c"
// = "b" | "c"
```

### 2.4 实际应用

```typescript
type Event = "click" | "focus" | "blur" | "scroll" | "resize";

// 鼠标事件
type MouseEvents = Extract<Event, "click">;
// "click"

// 非鼠标事件
type NonMouseEvents = Exclude<Event, "click">;
// "focus" | "blur" | "scroll" | "resize"

function handleMouseEvent(event: MouseEvents): void {
  console.log(`处理鼠标事件: ${event}`);
}

handleMouseEvent("click");
// 输出: 处理鼠标事件: click
```

## 三、注意事项与常见陷阱

1. **分布式条件类型**：`Exclude` 和 `Extract` 都利用分布式特性逐个处理联合成员
2. **非联合类型**：对非联合类型使用，结果要么是原类型要么是 `never`
3. **与 `NonNullable` 的关系**：`NonNullable<T>` 等价于 `Exclude<T, null | undefined>`
4. **性能良好**：联合类型成员数量合理时性能无问题
