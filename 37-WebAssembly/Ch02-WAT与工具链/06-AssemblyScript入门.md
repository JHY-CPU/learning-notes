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

## 五、高级类型和泛型

```typescript
// assembly/advanced.ts

// 使用静态数组（比普通数组更高效）
export function processStaticArray(): StaticArray<i32> {
  const arr = StaticArray.fromArray<i32>([1, 2, 3, 4, 5]);
  let sum: i32 = 0;
  for (let i = 0; i < arr.length; i++) {
    sum += unchecked(arr[i]);  // 跳过边界检查（更快但不安全）
  }
  return arr;
}

// 泛型函数
export function map<T, U>(
  arr: StaticArray<T>,
  fn: (value: T, index: i32) => U
): StaticArray<U> {
  const result = new StaticArray<U>(arr.length);
  for (let i = 0; i < arr.length; i++) {
    unchecked(result[i]) = fn(unchecked(arr[i]), i);
  }
  return result;
}

// 使用泛型
export function doubleValues(arr: StaticArray<f64>): StaticArray<f64> {
  return map<f64, f64>(arr, (v, _i) => v * 2.0);
}

// 位运算技巧
export function countBits(n: u32): u32 {
  let count: u32 = 0;
  while (n != 0) {
    count += n & 1;
    n >>= 1;
  }
  return count;
}
```

## 六、与 JavaScript 互操作的进阶用法

```typescript
// assembly/interop.ts

// 导入 JavaScript 函数
@external("env", "js_log")
declare function jsLog(value: i32): void;

@external("env", "js_random")
declare function jsRandom(): f64;

// 使用 @unmanaged 裁剪 GC 开销
@unmanaged
class Point {
  x: f64;
  y: f64;
}

// 使用 IDL 类型与 JavaScript 交互
@inline
export function createPoint(x: f64, y: f64): usize {
  const ptr = memory.allocate(16);  // sizeof(Point) = 2 * 8 bytes
  store<f64>(ptr, x, 0);
  store<f64>(ptr, y, 8);
  return ptr;
}

@inline
export function getX(ptr: usize): f64 {
  return load<f64>(ptr, 0);
}
```

```javascript
// JavaScript 端
import { add, factorial, processStaticArray } from './build/release.js';

// 读取 AssemblyScript 导出的 StaticArray
function readStaticArray(ptr) {
  // AssemblyScript 内存布局：
  // [4 bytes: GC header][4 bytes: length][... elements]
  const memory = add.__get ArrayBuffer();
  const view = new DataView(memory);
  const length = view.getInt32(ptr - 4, true);
  const result = [];
  for (let i = 0; i < length; i++) {
    result.push(view.getInt32(ptr + i * 4, true));
  }
  return result;
}

console.log(factorial(10)); // 3628800
console.log(add(100, 200)); // 300
```

## 七、性能优化技巧

```bash
# 构建优化
npm run asbuild:release

# 使用 asbuild 配置文件
# asconfig.json
{
  "targets": {
    "debug": {
      "outFile": "build/debug.wasm",
      "textFile": "build/debug.wat",
      "sourceMap": true,
      "debug": true
    },
    "release": {
      "outFile": "build/release.wasm",
      "textFile": "build/release.wat",
      "sourceMap": false,
      "optimizeLevel": 3,
      "shrinkLevel": 0,
      "converge": true,
      "noAssert": true
    }
  },
  "options": {
    "importMemory": false,
    "exportMemory": true,
    "memoryBase": 0
  }
}
```

## 八、AssemblyScript vs 其他方案对比

| 特性 | AssemblyScript | Rust+wasm-pack | Emscripten |
|------|---------------|----------------|------------|
| 学习曲线 | 低（TS 语法） | 中等 | 中高（C/C++） |
| 二进制大小 | 最小 | 小 | 中等 |
| 执行速度 | 快 | 最快 | 最快 |
| 生态系统 | 较小 | 丰富 | 最丰富 |
| 适用场景 | 简单计算、脚本 | 系统级、性能关键 | 遗留代码移植 |
| 内存管理 | 手动/GC | 手动/RAII | 手动 |
| 异步支持 | 有限 | 良好 | 良好 |
