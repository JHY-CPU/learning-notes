# 调试 TypeScript

## 一、概念说明

调试 TypeScript 需要 **Source Map** 支持，它建立了编译后 JS 与原始 TS 之间的映射关系。开启 source map 后，可以在 VS Code 或 Chrome DevTools 中直接在 `.ts` 源码上设置断点、查看变量。

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

## 三、注意事项与常见陷阱

1. **sourceMap vs inlineSourceMap**：生产环境不应包含 source map
2. **路径映射调试**：使用 `paths` 时可能需要 `resolveSourceMapLocations`
3. **`ts-node` 调试**：使用 `ts-node` 时设置 `runtimeArgs: ["-r", "ts-node/register"]`
4. **断点不命中**：检查 `outFiles` 路径是否正确指向编译输出
