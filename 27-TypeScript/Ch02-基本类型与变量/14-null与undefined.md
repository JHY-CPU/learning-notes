# null 与 undefined

## 一、概念说明

`null` 表示"有意地没有值"，`undefined` 表示"未定义"。在 `strictNullChecks` 模式下，`null` 和 `undefined` 不能赋值给其他类型，必须显式使用联合类型（如 `string | null`）才能表示可空值。这大幅减少了空指针错误，是 TypeScript 最重要的安全特性之一。

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

### 2.4 与 JavaScript 的对比

```javascript
// JavaScript —— 空指针运行时才崩溃
function getUserName(user) {
  return user.profile.name; // 如果 user.profile 是 undefined，崩溃
}
getUserName({}); // TypeError: Cannot read property 'name' of undefined
```

```typescript
// TypeScript —— 编译时发现问题
interface User {
  profile?: { name: string };
}

function getUserName(user: User): string {
  // return user.profile.name; // ❌ 编译错误: 'profile' 可能是 undefined
  return user.profile?.name ?? "匿名用户"; // ✅ 安全处理
}

console.log(getUserName({})); // 输出: 匿名用户
```

### 2.5 空值合并赋值 `??=`

```typescript
let config: { timeout?: number } = {};

// ??= 只在值为 null/undefined 时赋值
config.timeout ??= 5000;
console.log(config.timeout); // 输出: 5000

config.timeout ??= 10000; // 已有值，不覆盖
console.log(config.timeout); // 输出: 5000
```

## 三、注意事项与常见陷阱

1. **始终开启 `strictNullChecks`**：这是最重要的严格检查选项，应始终启用
2. **`?.` 短路求值**：`a?.b.c` 中如果 `a` 是 `null`，整个表达式返回 `undefined`，不会访问 `.c`
3. **`??` vs `||`**：`??` 只检查 `null`/`undefined`，`||` 检查所有假值（包括 `0`、`""`、`false`）
4. **可选参数的默认值**：`function f(x?: string)` 等价于 `x: string | undefined`
5. **`void` 与 `undefined`**：非严格模式下 `void` 函数可以返回 `undefined`，严格模式下不同
6. **`undefined` 是变量的默认值**：未初始化的变量和缺失的对象属性值为 `undefined`
