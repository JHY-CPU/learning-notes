# Branded/Tagged 类型

## 一、概念说明

Branded 类型（也称 Tagged 类型）通过添加私有的"品牌"属性，创建**名义类型**（Nominal Type）。TypeScript 是**结构类型**系统，两个结构相同的类型可以互相赋值。Branded 类型通过附加唯一的品牌标记来阻止这种行为，模拟名义类型的语义。典型应用：区分 UserId 和 ProductId、确保邮箱经过验证等。

## 二、具体用法

### 2.1 基本 Branded 类型定义

```typescript
// 品牌类型构造
type Brand<T, B extends string> = T & { readonly __brand: B };

// 定义不同 ID 类型
type UserId = Brand<number, "UserId">;
type ProductId = Brand<number, "ProductId">;
type OrderId = Brand<string, "OrderId">;
```

### 2.2 构造函数（品牌化入口）

```typescript
// 构造函数将原始类型转为 Branded 类型
const UserId = (id: number): UserId => id as UserId;
const ProductId = (id: number): ProductId => id as ProductId;

// 使用
function getUser(id: UserId): string {
  return `用户 ${id}`;
}

const userId = UserId(1);
const productId = ProductId(42);

getUser(userId);      // OK
// getUser(productId); // 编译错误：ProductId 不兼容 UserId
// getUser(1);         // 编译错误：number 不兼容 UserId
```

### 2.3 验证后的 Branded 类型

```typescript
type Email = Brand<string, "Email">;
type PositiveNumber = Brand<number, "PositiveNumber">;

function createEmail(input: string): Email {
  if (!input.includes("@")) {
    throw new Error("无效邮箱格式");
  }
  return input as Email;
}

function createPositive(n: number): PositiveNumber {
  if (n <= 0) throw new Error("必须为正数");
  return n as PositiveNumber;
}

function sendEmail(to: Email, subject: string): void {
  // 确保 to 一定是经过验证的邮箱
  console.log(`发送到 ${to}: ${subject}`);
}

const email = createEmail("user@example.com");
sendEmail(email, "欢迎"); // OK
// sendEmail("not-email", "测试"); // 编译错误
```

### 2.4 货币单位防混淆

```typescript
type Currency<T extends string> = Brand<number, T>;
type USD = Currency<"USD">;
type EUR = Currency<"EUR">;
type CNY = Currency<"CNY">;

const usd = (n: number): USD => n as USD;
const eur = (n: number): EUR => n as EUR;

function addUSD(a: USD, b: USD): USD {
  return (a + b) as USD;
}

const price = usd(99.99);
const tax = usd(8.50);
const total = addUSD(price, tax); // OK
// addUSD(price, eur(10)); // 编译错误：EUR 不兼容 USD
```

### 2.5 配合泛型约束

```typescript
// 类型安全的数据库查询
function findById<T extends Brand<number, string>>(
  table: string,
  id: T
): Promise<any> {
  return db.query(`SELECT * FROM ${table} WHERE id = $1`, [id]);
}

// 不同表用不同 Branded ID
type UserId = Brand<number, "UserId">;
type PostId = Brand<number, "PostId">;

findById("users", UserId(1));   // OK
findById("posts", PostId(1));   // OK
// findById("users", PostId(1)); // 可能通过（如果泛型约束不严格）
```

### 2.6 与 JavaScript 的对比

```javascript
// JavaScript：无类型保护，ID 可以随意混用
function getUser(id) { return `用户 ${id}`; }
function getProduct(id) { return `产品 ${id}`; }

getUser(1);
getProduct(1);
getUser("abc"); // 运行时可能出错，但无编译提示

// TypeScript Branded：编译时防止混用
// getUser(ProductId(42)); // 编译错误
```

## 三、注意事项与常见陷阱

1. **品牌属性是编译时的**：`__brand` 在运行时不存在，不影响对象属性
2. **类型断言是入口**：必须通过包含 `as Brand` 的构造函数创建，这是唯一"不安全"的入口
3. **不能直接赋值**：普通 `number` 不能直接赋给 `UserId`，必须通过构造函数
4. **适用于 ID、货币、单位等**：防止语义相同但含义不同的值混淆
5. **品牌属性名要独特**：用 `__brand` 等不常见的名字避免与业务属性冲突
6. **readonly 防止篡改**：品牌属性标记为 `readonly` 确保不被修改
