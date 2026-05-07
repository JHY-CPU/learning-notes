# 编译选项 strict 模式

## 一、概念说明

`strict` 是 TypeScript 中最重要的编译选项，设为 `true` 时会启用所有严格的类型检查。它是一个"总开关"，开启后等价于同时启用 `noImplicitAny`、`strictNullChecks`、`strictFunctionTypes` 等多个子选项。强烈建议所有新项目开启此选项。

## 二、具体用法

### 2.1 开启 strict 模式

```json
// tsconfig.json
{
  "compilerOptions": {
    "strict": true
  }
}
```

等价于手动开启以下全部选项：

```json
{
  "compilerOptions": {
    "noImplicitAny": true,           // 禁止隐式 any
    "strictNullChecks": true,        // 严格空值检查
    "strictFunctionTypes": true,     // 严格函数类型检查
    "strictBindCallApply": true,     // 严格 bind/call/apply
    "strictPropertyInitialization": true, // 严格属性初始化
    "noImplicitThis": true,          // 禁止隐式 this
    "alwaysStrict": true             // 始终使用严格模式
  }
}
```

### 2.2 noImplicitAny 示例

```typescript
// strict 模式下，参数必须有类型注解或能被推断

// ❌ 错误：隐式 any
// function add(a, b) { return a + b; }
// 编译错误: Parameter 'a' implicitly has an 'any' type.

// ✅ 正确：显式类型注解
function add(a: number, b: number): number {
  return a + b;
}

console.log(add(3, 5));
// 输出: 8
```

### 2.3 strictNullChecks 示例

```typescript
// 开启后，null 和 undefined 不能赋值给其他类型

// ❌ 错误
// let name: string = null;
// 编译错误: Type 'null' is not assignable to type 'string'.

// ✅ 正确：使用联合类型
let name: string | null = null;
name = "张三";

// 可选链安全访问
const user: { name?: string } = {};
console.log(user.name?.toUpperCase());
// 输出: undefined
```

## 三、注意事项与常见陷阱

1. **新项目必须开启**：没有 `strict` 的 TypeScript 只是"加了注释的 JavaScript"
2. **迁移老项目**：可逐步开启子选项而非一次性开启 `strict`
3. **`strictNullChecks` 影响最大**：会显著增加需要处理的空值情况
4. **避免用 `!` 绕过**：非空断言（`!`）会跳过检查，应谨慎使用
