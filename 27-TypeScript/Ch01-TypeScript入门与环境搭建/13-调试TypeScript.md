# 调试 TypeScript

## 一、概念说明

调试 TypeScript 需要 **Source Map** 支持，它建立了编译后 JS 与原始 TS 之间的映射关系。开启 source map 后，可以在 VS Code 或 Chrome DevTools 中直接在 `.ts` 源码上设置断点、查看变量。Source Map 是 TypeScript 开发中调试体验的关键，没有它你只能调试编译后的 JS 代码。

## 二、具体用法

### 2.1 开启 Source Map

```json
// tsconfig.json
{
  "compilerOptions": {
    "sourceMap": true,
    "outDir": "./dist"
  }
}
```

### 2.2 VS Code 调试配置

```json
// .vscode/launch.json
{
  "version": "0.2.0",
  "configurations": [
    {
      "type": "node",
      "request": "launch",
      "name": "调试 TypeScript",
      "program": "${workspaceFolder}/src/index.ts",
      "preLaunchTask": "tsc: build",
      "outFiles": ["${workspaceFolder}/dist/**/*.js"],
      "sourceMaps": true
    }
  ]
}
```

### 2.3 在代码中设置断点

```typescript
// src/index.ts
function calculateTotal(prices: number[]): number {
  let total = 0;                    // <-- 在此行设置断点 (F9)
  for (const price of prices) {
    total += price;                 // 调试时可观察 total 的变化
  }
  return total;
}

const prices = [29.99, 15.50, 8.75];
const result = calculateTotal(prices);
console.log(`总计: ¥${result.toFixed(2)}`);
// 输出: 总计: ¥54.24
```

**调试过程：**
```
1. 按 F5 启动调试
2. 程序在断点处暂停
3. 在"变量"面板查看 total、price 的值
4. 按 F10 逐步执行，F5 继续运行
```

### 2.4 调试 Express 应用

```json
// .vscode/launch.json —— 调试 Node.js 服务器
{
  "version": "0.2.0",
  "configurations": [
    {
      "type": "node",
      "request": "launch",
      "name": "调试 Express 服务器",
      "runtimeArgs": ["-r", "ts-node/register"],
      "args": ["${workspaceFolder}/src/server.ts"],
      "sourceMaps": true
    }
  ]
}
```

```typescript
// src/server.ts
import express from "express";

const app = express();

app.get("/api/users/:id", (req, res) => {
  const userId = parseInt(req.params.id); // 在此行打断点
  console.log(`请求用户: ${userId}`);
  res.json({ id: userId, name: "张三" });
});

app.listen(3000, () => {
  console.log("服务器运行在 http://localhost:3000");
});
```

### 2.5 Chrome DevTools 调试

```bash
# 使用 --inspect 启动 Node.js
node --inspect -r ts-node/register src/index.ts

# 然后在 Chrome 中打开 chrome://inspect
```

## 三、注意事项与常见陷阱

1. **sourceMap vs inlineSourceMap**：生产环境不应包含 source map，避免暴露源码
2. **路径映射调试**：使用 `paths` 时可能需要 `resolveSourceMapLocations`
3. **`ts-node` 调试**：使用 `ts-node` 时设置 `runtimeArgs: ["-r", "ts-node/register"]`
4. **断点不命中**：检查 `outFiles` 路径是否正确指向编译输出
5. **断点行偏移**：如果断点行数对不上，检查 `sourceMap` 是否正确生成
6. **Vite 调试**：Vite 项目开箱即用支持 source map，直接在浏览器 DevTools 中调试即可
