# Opaque 类型

## 一、概念说明

Opaque 类型（不透明类型）与 Branded 类型类似，通过隐藏底层类型创建新类型。外部代码只知道它是某种类型，但不知道内部结构，强制通过特定 API 操作。

## 二、具体用法

### 2.1 基本 Opaque 类型

```typescript
// Opaque 类型定义
type Opaque<Type, Token extends string> = Type & {
  readonly __opaque__: Token;
};

type USD = Opaque<number, "USD">;
type EUR = Opaque<number, "EUR">;

function createUSD(amount: number): USD {
  return amount as USD;
}

function addUSD(a: USD, b: USD): USD {
  return (a + b) as USD;
}

const price = createUSD(99.99);
const tax = createUSD(8.50);
const total = addUSD(price, tax);

console.log(`总计: $${total}`);
// 输出: 总计: $108.49
// addUSD(price, 10 as EUR); // ❌ EUR 不兼容 USD
```

### 2.2 封装内部状态

```typescript
type EncryptedData = Opaque<string, "Encrypted">;

function encrypt(data: string): EncryptedData {
  return Buffer.from(data).toString("base64") as EncryptedData;
}

function decrypt(data: EncryptedData): string {
  return Buffer.from(data, "base64").toString();
}

const secret = encrypt("敏感数据");
console.log(decrypt(secret));
// 输出: 敏感数据
```

## 三、注意事项与常见陷阱

1. **与 Branded 类似**：区别在于 Opaque 强调"隐藏实现"
2. **必须通过工厂函数创建**：确保数据有效性
3. **编译时安全性**：运行时仍可访问底层值
4. **API 设计模式**：限制用户只能通过特定函数操作
