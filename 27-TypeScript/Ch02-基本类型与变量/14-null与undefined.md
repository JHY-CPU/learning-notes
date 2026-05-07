# null 与 undefined

## 一、概念说明

`null` 表示"有意地没有值"，`undefined` 表示"未定义"。在 `strictNullChecks` 模式下，`null` 和 `undefined` 不能赋值给其他类型，必须显式使用联合类型（如 `string | null`）才能表示可空值。这大幅减少了空指针错误。

## 二、具体用法

### 2.1 strictNullChecks 行为

```typescript
// 开启 strictNullChecks 后
let name: string = "Alice";
// name = null; // ❌ 编译错误

// 必须显式声明可空
let nickname: string | null = null;
nickname = "小明";

console.log(nickname);
// 输出: 小明
```

### 2.2 可选链 `?.`

```typescript
interface User {
  name: string;
  address?: {
    city?: string;
    street?: string;
  };
}

const user: User = { name: "张三" };

// 安全访问嵌套属性
const city = user.address?.city;
console.log(city);
// 输出: undefined

// 安全调用方法
const upperCity = user.address?.city?.toUpperCase();
console.log(upperCity);
// 输出: undefined
```

### 2.3 空值合并 `??`

```typescript
// ?? 只在 null/undefined 时使用默认值
const input: string | null = null;
const value = input ?? "默认值";
console.log(value);
// 输出: 默认值

// 与 || 的区别：|| 会把 ""、0、false 也当作假值
const emptyStr = "" || "fallback"; // "fallback"
const emptyStr2 = "" ?? "fallback"; // ""（空字符串不是 null/undefined）

console.log(emptyStr, emptyStr2);
// 输出: fallback ""
```

## 三、注意事项与常见陷阱

1. **始终开启 `strictNullChecks`**：这是最重要的严格检查选项
2. **`?.` 短路求值**：`a?.b.c` 中如果 `a` 是 `null`，整个表达式返回 `undefined`
3. **`??` vs `||`**：`??` 只检查 `null`/`undefined`，`||` 检查所有假值
4. **可选参数的默认值**：`function f(x?: string)` 等价于 `x: string | undefined`
