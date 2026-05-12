# Opaque 类型

## 一、概念说明

Opaque 类型（不透明类型）与 Branded 类型类似，通过隐藏底层类型来创建新类型。外部代码只知道它是某种类型，但**不能直接访问或构造**，必须通过特定的工厂函数（API）操作。Opaque 类型强调**封装性**——隐藏实现细节，强制用户通过受控接口交互。常用于加密数据、令牌、经验证的值等。

## 二、具体用法

### 2.1 基本 Opaque 类型定义

```typescript
// Opaque 类型：隐藏 Token 使其不对外暴露
type Opaque<Type, Token extends string> = Type & {
  readonly __opaque__: Token;
};

// 定义 Opaque 类型
type USD = Opaque<number, "USD">;
type EUR = Opaque<number, "EUR">;
type EncryptedData = Opaque<string, "Encrypted">;
type ValidatedEmail = Opaque<string, "ValidatedEmail">;
```

### 2.2 工厂函数（受控入口）

```typescript
function createUSD(amount: number): USD {
  if (amount < 0) throw new Error("金额不能为负");
  return amount as USD;
}

function addUSD(a: USD, b: USD): USD {
  return (a + b) as USD;
}

function formatUSD(amount: USD): string {
  return `$${amount.toFixed(2)}`;
}

const price = createUSD(99.99);
const tax = createUSD(8.50);
const total = addUSD(price, tax);

console.log(formatUSD(total)); // $108.49
// createUSD(-5);              // 运行时错误
// addUSD(price, 10 as EUR);  // 编译错误
```

### 2.3 封装加密数据

```typescript
function encrypt(data: string): EncryptedData {
  // 简化示例，实际使用真正的加密算法
  return Buffer.from(data).toString("base64") as EncryptedData;
}

function decrypt(data: EncryptedData): string {
  return Buffer.from(data, "base64").toString();
}

// 使用方只能通过 encrypt 获取 EncryptedData
const secret = encrypt("敏感信息");
console.log(decrypt(secret)); // 敏感信息

// 以下被阻止
// decrypt("plain text"); // 编译错误：string 不兼容 EncryptedData
```

### 2.4 API 令牌封装

```typescript
type ApiToken = Opaque<string, "ApiToken">;

function authenticate(credentials: { username: string; password: string }): ApiToken {
  // 验证逻辑
  const token = Buffer.from(`${credentials.username}:${Date.now()}`).toString("base64");
  return token as ApiToken;
}

function makeRequest(token: ApiToken, endpoint: string): Promise<any> {
  return fetch(endpoint, {
    headers: { Authorization: `Bearer ${token}` },
  });
}

const token = authenticate({ username: "admin", password: "secret" });
makeRequest(token, "/api/data"); // OK
// makeRequest("fake-token", "/api/data"); // 编译错误
```

### 2.5 与 Branded 类型的区别

```typescript
// Branded：品牌属性暴露在类型定义中
type BrandedId = number & { __brand: "UserId" };
// 可以看到底层是 number，品牌是 "UserId"

// Opaque：Token 参数完全隐藏
type OpaqueId = Opaque<number, "SecretToken">;
// 只知道是 number 的子类型，Token 具体值不可见

// Opaque 更适合"黑盒"场景：加密数据、令牌
// Branded 更适合"标记"场景：区分不同 ID 类型
```

### 2.6 与 JavaScript 的对比

```javascript
// JavaScript：无编译时封装
function createToken(userId) {
  return `token_${userId}_${Date.now()}`; // 返回 string
}
const token = createToken(1);
// 任何 string 都可以传给需要 token 的函数
// const fake = "not-a-token";
// makeRequest(fake); // 运行时才可能发现错误

// TypeScript Opaque：编译时强制封装
// 只能通过 createToken 获取 ApiToken 类型
// 普通 string 无法替代 ApiToken
```

## 三、注意事项与常见陷阱

1. **与 Branded 的区别**：Opaque 强调"隐藏实现"，Branded 强调"标记区分"。两者可以互换使用
2. **工厂函数是唯一入口**：必须通过函数创建 Opaque 类型，这是类型安全的关键
3. **编译时安全性，运行时透明**：运行时仍可通过类型断言或 `any` 绕过
4. **API 设计模式**：限制用户只能通过导出的函数操作 Opaque 类型
5. **Token 可以是任意字符串**：只是编译时标记，不影响运行时行为
6. **不暴露底层类型**：Opaque 类型的使用者不应知道底层实现（如 number 或 string）
