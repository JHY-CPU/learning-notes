# Branded/Tagged 类型

## 一、概念说明

Branded 类型（也称 Tagged 类型）通过添加私有"品牌"属性，创建名义类型（Nominal Type），防止不同类型之间意外混用。TypeScript 是结构类型的，Branded 类型模拟名义类型的行为。

## 二、具体用法

### 2.1 基本 Branded 类型

```typescript
// 品牌类型定义
type Brand<T, B extends string> = T & { __brand: B };

type UserId = Brand<number, "UserId">;
type ProductId = Brand<number, "ProductId">;

// 构造函数
const UserId = (id: number): UserId => id as UserId;
const ProductId = (id: number): ProductId => id as ProductId;

function getUser(id: UserId): string {
  return `用户 ${id}`;
}

const userId = UserId(1);
const productId = ProductId(1);

console.log(getUser(userId));     // ✅
// getUser(productId);             // ❌ 类型不兼容
// getUser(1);                     // ❌ 普通 number 不兼容
```

### 2.2 验证后的类型

```typescript
type Email = Brand<string, "Email">;

function createEmail(input: string): Email {
  if (!input.includes("@")) {
    throw new Error("无效的邮箱格式");
  }
  return input as Email;
}

function sendEmail(to: Email, subject: string): void {
  console.log(`发送到 ${to}: ${subject}`);
}

const email = createEmail("user@example.com");
sendEmail(email, "欢迎");
// 输出: 发送到 user@example.com: 欢迎
```

## 三、注意事项与常见陷阱

1. **品牌属性是编译时的**：运行时不存在
2. **类型断言是入口**：需要通过验证函数创建
3. **不能直接赋值**：`number` 不能直接赋给 `UserId`
4. **适用于 ID、货币、单位等**：防止类型混淆
