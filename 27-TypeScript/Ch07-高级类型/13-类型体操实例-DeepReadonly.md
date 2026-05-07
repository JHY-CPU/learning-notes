# 类型体操实例 - DeepReadonly

## 一、概念说明

实现一个递归的 `DeepReadonly<T>` 工具类型，使对象的所有层级属性都变为 `readonly`。这是经典的类型编程练习。

## 二、具体用法

### 2.1 实现

```typescript
type DeepReadonly<T> = T extends (...args: any[]) => any
  ? T
  : T extends object
    ? { readonly [K in keyof T]: DeepReadonly<T[K]> }
    : T;
```

### 2.2 使用

```typescript
interface Config {
  server: {
    host: string;
    port: number;
    ssl: { enabled: boolean; cert: string };
  };
  db: { url: string };
}

type FrozenConfig = DeepReadonly<Config>;

const config: FrozenConfig = {
  server: {
    host: "localhost",
    port: 3000,
    ssl: { enabled: true, cert: "/path/to/cert" },
  },
  db: { url: "postgres://localhost/db" },
};

// config.server.host = "other";     // ❌ readonly
// config.server.ssl.enabled = false; // ❌ 深层 readonly

console.log(config.server.ssl.enabled);
// 输出: true
```

### 2.3 排除函数

```typescript
// 函数保持不变，只冻结数据对象
type Handler = () => void;
type FrozenHandler = DeepReadonly<Handler>; // () => void（不变）
```

## 三、注意事项与常见陷阱

1. **函数应跳过**：函数类型不应被 readonly 化
2. **数组处理**：`DeepReadonly<T[]>` 产生 `readonly DeepReadonly<T>[]`
3. **递归深度**：TypeScript 对递归有限制
4. **性能**：大型嵌套类型可能影响编译速度
