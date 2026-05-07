# 类型体操实例 - DeepPartial

## 一、概念说明

实现递归的 `DeepPartial<T>`，使对象的所有层级属性都变为可选。与内置 `Partial<T>`（浅层）不同，`DeepPartial` 递归处理嵌套对象。

## 二、具体用法

### 2.1 实现

```typescript
type DeepPartial<T> = T extends (...args: any[]) => any
  ? T
  : T extends object
    ? { [K in keyof T]?: DeepPartial<T[K]> }
    : T;
```

### 2.2 使用

```typescript
interface Config {
  server: { host: string; port: number; };
  db: { url: string; pool: number; };
  logging: { level: "debug" | "info" | "warn"; file: string; };
}

// 所有层级都可选
const partial: DeepPartial<Config> = {
  server: { host: "localhost" },
  db: { pool: 10 },
};

console.log(partial);
// 输出: { server: { host: "localhost" }, db: { pool: 10 } }
```

### 2.3 应用：合并配置

```typescript
function mergeConfig(
  defaults: Config,
  overrides: DeepPartial<Config>
): Config {
  // 深度合并逻辑
  return { ...defaults, ...overrides } as Config;
}
```

## 三、注意事项与常见陷阱

1. **数组处理**：`DeepPartial<T[]>` 使数组元素属性可选
2. **可选 vs undefined**：`?` 使属性可选，值类型不变
3. **递归深度限制**：TypeScript 约 1000 层
4. **配合 Required 反向**：`DeepRequired<DeepPartial<T>>` 恢复必选
