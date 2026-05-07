# 动态导入 import()

## 一、概念说明

动态导入使用 `import()` 函数在运行时按需加载模块，支持代码分割和懒加载。返回 `Promise<Module>`。

## 二、具体用法

### 2.1 基本动态导入

```typescript
// 按需加载模块
async function loadMath() {
  const math = await import("./math.js");
  console.log(math.add(1, 2));
  // 输出: 3
}

loadMath();
```

### 2.2 条件加载

```typescript
async function getEnvConfig() {
  if (process.env.NODE_ENV === "production") {
    return await import("./config.prod.js");
  }
  return await import("./config.dev.js");
}
```

### 2.3 类型安全的动态导入

```typescript
type MathModule = typeof import("./math.js");

async function loadMath(): Promise<MathModule> {
  return await import("./math.js");
}

(async () => {
  const math = await loadMath();
  console.log(math.add(5, 3));
  // 输出: 8
})();
```

## 三、注意事项与常见陷阱

1. **返回 Promise**：必须 `await` 或 `.then()`
2. **类型导入不能动态**：`import("./types.js")` 只获取值
3. **代码分割**：构建工具会自动分割动态导入的模块
4. **错误处理**：动态导入可能因网络等原因失败
