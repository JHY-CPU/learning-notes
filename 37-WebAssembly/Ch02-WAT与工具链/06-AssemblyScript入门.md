# AssemblyScript入门

## 一、概念说明

AssemblyScript 是 TypeScript 的子集，可直接编译为 WASM，无需 C/C++ 知识。

```bash
# 安装 AssemblyScript
npm install --save-dev assemblyscript
npx asinit .
```

## 二、具体用法

### 2.1 基本语法

```typescript
// assembly/index.ts
export function add(a: i32, b: i32): i32 {
  return a + b;
}

export function factorial(n: i32): i32 {
  if (n <= 1) return 1;
  return n * factorial(n - 1);
}

// 使用数组
export function sum(arr: Int32Array): i32 {
  let total: i32 = 0;
  for (let i = 0; i < arr.length; i++) {
    total += arr[i];
  }
  return total;
}
```

### 2.2 内存操作

```typescript
// 分配内存
export function allocate(size: i32): usize {
  return heap.alloc(size);
}

// 释放内存
export function deallocate(ptr: usize): void {
  heap.free(ptr);
}

// 字符串操作
export function getString(): string {
  return "Hello, AssemblyScript!";
}
```

### 2.3 构建和运行

```bash
# 构建
npm run asbuild

# 构建优化版本
npm run asbuild:release

# 测试
npm test
```

### 2.4 JavaScript 集成

```javascript
import { add, factorial } from './build/release.js';

const result = add(1, 2);
console.log(result); // 3

const fact = factorial(5);
console.log(fact); // 120
```

## 三、注意事项与常见陷阱

1. **类型限制**：AssemblyScript 类型有限制
2. **JavaScript API**：不能直接使用所有 JavaScript API
3. **内存管理**：需要手动管理内存
4. **调试困难**：调试不如 JavaScript 方便
5. **性能**：某些操作可能不如 C/C++ 高效
