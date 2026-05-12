# 类型体操实例 - DeepPartial

## 一、概念说明

实现递归的 `DeepPartial<T>` 工具类型，使对象的**所有层级**属性都变为可选。与内置 `Partial<T>`（仅浅层可选）不同，`DeepPartial` 递归处理嵌套对象，适用于深度合并配置、部分更新 API 等场景。它是类型编程中最实用的工具类型之一。

## 二、具体用法

### 2.1 基础实现

```typescript
type DeepPartial<T> = T extends (...args: any[]) => any
  ? T
  : T extends object
    ? { [K in keyof T]?: DeepPartial<T[K]> }
    : T;
```

**实现要点**：
- 排除函数类型（函数不应被 Partial 化）
- 对对象类型添加 `?` 并递归处理属性
- 基本类型直接返回

### 2.2 使用示例

```typescript
interface AppConfig {
  server: {
    host: string;
    port: number;
    ssl: { enabled: boolean; cert: string };
  };
  database: {
    url: string;
    pool: { min: number; max: number };
  };
  logging: {
    level: "debug" | "info" | "warn";
    file: string;
  };
}

// 所有层级都可选
const overrides: DeepPartial<AppConfig> = {
  server: { port: 8080 },            // host 和 ssl 可以不提供
  database: { pool: { max: 50 } },   // url 和 min 可以不提供
  // logging 完全省略
};
```

### 2.3 深度合并配置

```typescript
function deepMerge<T>(target: T, source: DeepPartial<T>): T {
  const result = { ...target };
  for (const key of Object.keys(source) as (keyof T)[]) {
    const sourceVal = source[key];
    const targetVal = target[key];
    if (
      sourceVal !== null &&
      typeof sourceVal === "object" &&
      !Array.isArray(sourceVal) &&
      typeof targetVal === "object" &&
      targetVal !== null
    ) {
      (result as any)[key] = deepMerge(targetVal, sourceVal as any);
    } else if (sourceVal !== undefined) {
      (result as any)[key] = sourceVal;
    }
  }
  return result;
}

const defaults: AppConfig = {
  server: { host: "localhost", port: 3000, ssl: { enabled: false, cert: "" } },
  database: { url: "postgres://localhost/db", pool: { min: 2, max: 10 } },
  logging: { level: "info", file: "app.log" },
};

const config = deepMerge(defaults, {
  server: { port: 8080, ssl: { enabled: true } },
  logging: { level: "debug" },
});
// 结果保留默认值，只覆盖指定字段
```

### 2.4 对比 Partial 和 DeepPartial

```typescript
// 浅层 Partial：只让第一层可选
type ShallowPartialConfig = Partial<AppConfig>;
const shallow: ShallowPartialConfig = {
  server: { host: "localhost", port: 3000, ssl: { enabled: true, cert: "" } },
  // server 如果提供，必须包含所有字段（host、port、ssl 完整）
};

// 深层 DeepPartial：每个层级都可选
const deep: DeepPartial<AppConfig> = {
  server: { port: 8080 },  // 只提供 port，其他可以省略
};
```

### 2.5 配合 DeepRequired 反转

```typescript
type DeepRequired<T> = T extends (...args: any[]) => any
  ? T
  : T extends object
    ? { [K in keyof T]-?: DeepRequired<T[K]> }
    : T;

// 应用 DeepPartial 后再恢复
type PartialConfig = DeepPartial<AppConfig>;
type RestoredConfig = DeepRequired<PartialConfig>;
// RestoredConfig 等价于 AppConfig
```

### 2.6 与 JavaScript 的对比

```javascript
// JavaScript：运行时深度合并
const defaults = { server: { port: 3000, host: "localhost" }, db: { pool: 10 } };
const overrides = { server: { port: 8080 }, db: { pool: 20 } };

// lodash.merge 或手动实现
function deepMergeJS(target, source) {
  for (const key in source) {
    if (typeof source[key] === "object" && source[key] !== null) {
      target[key] = deepMergeJS(target[key] || {}, source[key]);
    } else {
      target[key] = source[key];
    }
  }
  return target;
}

// TypeScript DeepPartial：编译时保证类型安全
// 配合 deepMerge 函数，编译时就确保不传入无效字段
```

## 三、注意事项与常见陷阱

1. **数组特殊处理**：`DeepPartial<T[]>` 会使数组元素属性可选，但数组本身不会变为可选。可能需要特殊处理数组类型
2. **可选 vs undefined**：`?` 使属性可选（可以省略），但值类型不变。要允许 `undefined` 值需额外 `| undefined`
3. **递归深度限制**：TypeScript 约 1000 层递归，超深嵌套会编译失败
4. **联合类型处理**：联合类型中每个成员会分别被 Partial 化
5. **`never` 类型**：`DeepPartial<never>` 返回 `never`
6. **性能注意**：大型嵌套类型的 `DeepPartial` 可能显著影响编译速度
