# WASM生态系统

## 一、概念说明

WASM 生态系统包括工具链、运行时、框架等。

```bash
# 主要工具链
# Emscripten (C/C++)
# wasm-pack (Rust)
# AssemblyScript
# TinyGo
# wasi-sdk

# 运行时
# Wasmtime
# Wasmer
# WasmEdge
# wasm3
```

## 二、具体用法

### 2.1 开发工具

```bash
# wasm-pack (Rust)
wasm-pack build --target web
wasm-pack build --target nodejs

# Binaryen 优化工具
wasm-opt -O3 -o output.wasm input.wasm

# wasm-bindgen (Rust-JS 绑定)
# 自动生成 JavaScript 胶水代码

# wasm-snip (移除未使用代码)
wasm-snip input.wasm -o output.wasm
```

### 2.2 浏览器框架

```javascript
// wasm-bindgen 自动生成的 JavaScript
import init, { greet } from './pkg/my_lib.js';

async function main() {
  await init();
  greet('World');
}

main();

// Yew (Rust 前端框架)
// 使用 wasm-bindgen 与 JavaScript 交互
```

### 2.3 非浏览器运行时

```bash
# Wasmtime (Bytecode Alliance)
wasmtime run module.wasm

# Wasmer
wasmer run module.wasm

# WasmEdge (云原生)
wasmedge run module.wasm

# Node.js
node --experimental-wasm-modules app.js
```

### 2.4 WASI (WebAssembly System Interface)

```javascript
// WASI 提供系统接口
// 文件系统、网络、时钟等

(module
  (import "wasi_snapshot_preview1" "fd_write"
    (func $fd_write (param i32 i32 i32 i32) (result i32)))
  (import "wasi_snapshot_preview1" "clock_time_get"
    (func $clock_time_get (param i32 i64 i32) (result i32))))
```

## 三、注意事项与常见陷阱

1. **工具选择**：根据源语言选择工具链
2. **运行时差异**：不同运行时行为可能不同
3. **WASI 兼容性**：WASI 还在演进中
4. **包大小**：使用工具优化包大小
5. **版本兼容**：注意工具链版本兼容性
