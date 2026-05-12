# 动态导入 import()

## 一、概念说明

动态导入使用 `import()` 函数在**运行时**按需加载模块，返回 `Promise<Module>`。它支持代码分割（Code Splitting）和懒加载（Lazy Loading），是优化应用初始加载性能的重要手段。与静态导入不同，动态导入的路径可以是变量，实现按条件、按路由等灵活加载。

## 二、具体用法

### 2.1 基本动态导入

```typescript
// 按需加载模块
async function loadMath() {
  const math = await import("./math.js");
  console.log(math.add(1, 2)); // 3
  console.log(math.PI);        // 3.14159
}

loadMath();
```

### 2.2 条件加载

```typescript
// 根据环境加载不同配置
async function getEnvConfig() {
  if (process.env.NODE_ENV === "production") {
    return await import("./config.prod.js");
  }
  return await import("./config.dev.js");
}

// 根据用户选择加载语言包
async function loadLocale(lang: string) {
  const messages = await import(`./locales/${lang}.json`);
  return messages.default;
}
```

### 2.3 类型安全的动态导入

```typescript
// 使用 typeof import 获取模块类型
type MathModule = typeof import("./math.js");

async function loadMath(): Promise<MathModule> {
  return await import("./math.js");
}

// 使用
(async () => {
  const math = await loadMath();
  // math.add, math.PI 等都有完整类型
  const result: number = math.add(5, 3); // 类型安全
})();
```

### 2.4 React 组件懒加载

```typescript
import { lazy, Suspense } from "react";

// React.lazy 配合动态导入实现组件懒加载
const Dashboard = lazy(() => import("./pages/Dashboard.js"));
const Settings = lazy(() => import("./pages/Settings.js"));

function App() {
  return (
    <Suspense fallback={<div>加载中...</div>}>
      <Routes>
        <Route path="/dashboard" element={<Dashboard />} />
        <Route path="/settings" element={<Settings />} />
      </Routes>
    </Suspense>
  );
}
```

### 2.5 错误处理

```typescript
async function loadModule(name: string) {
  try {
    const module = await import(`./modules/${name}.js`);
    return module;
  } catch (error) {
    console.error(`加载模块 ${name} 失败:`, error);
    // 降级方案
    return await import("./modules/fallback.js");
  }
}
```

### 2.6 与 JavaScript 的对比

```javascript
// JavaScript：动态导入语法相同
const module = await import("./math.js");

// TypeScript：额外获得类型安全
// const math: typeof import("./math") = await import("./math.js");
// 返回值有完整类型信息，IDE 自动补全
```

## 三、注意事项与常见陷阱

1. **返回 Promise**：必须 `await` 或 `.then()` 获取模块内容
2. **路径中的变量**：动态路径（如 ``import(`./${name}.js`)``）无法被构建工具静态分析，不会被分割
3. **类型安全**：使用 `typeof import()` 获取模块类型以保持类型安全
4. **错误处理**：网络失败、路径错误等会导致 Promise reject，务必捕获
5. **代码分割**：构建工具（Webpack、Vite）会自动将动态导入的模块分割为独立 chunk
6. **CommonJS 中的动态导入**：`module` 设为 `CommonJS` 时，`import()` 会编译为 `Promise.resolve().then(() => require(...))`
