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

## 五、主流框架和库

| 框架/库 | 语言 | 用途 | 链接 |
|---------|------|------|------|
| Yew | Rust | 前端 UI 框架 | yew.rs |
| Leptos | Rust | 响应式前端框架 | leptos.dev |
| Bevy | Rust | 游戏引擎（可导出 WASM） | bevyengine.org |
| Emscripten | C/C++ | 完整的编译工具链 | emscripten.org |
| Blazor WebAssembly | C# | .NET 前端框架 | dotnet.microsoft.com |
| Pyodide | Python | Python 科学计算栈 | pyodide.org |
| TeaVM | Java/Kotlin | JVM 到 WASM | teavm.org |
| Wasmtime | 多语言 | 高性能 WASM 运行时 | wasmtime.dev |

## 六、WASI 运行时对比

```bash
# Wasmtime（Bytecode Alliance 官方）
wasmtime --dir=. module.wasm           # 挂载当前目录
wasmtime --env KEY=VALUE module.wasm   # 设置环境变量

# Wasmer（多后端支持）
wasmer run module.wasm
wasmer run --backend=cranelift module.wasm  # 选择编译后端

# WasmEdge（云原生优化）
wasmedge --dir=. module.wasm
wasmedge --reactor module.wasm func_name    # 调用特定函数

# Node.js 原生支持（实验性）
node --experimental-wasm-modules app.mjs
```

## 七、包大小优化工具链

```bash
# 1. wasm-opt（Binaryen）：优化 WASM 字节码
wasm-opt -Os -o output.wasm input.wasm

# 2. wasm-snip：移除未使用的函数
wasm-snip input.wasm -o output.wasm --snip-func "unused_func"

# 3. wasm-strip：移除调试信息和自定义段
wasm-strip input.wasm -o output.wasm

# 4. LTO（Link Time Optimization）：Rust 编译时启用
# Cargo.toml:
# [profile.release]
# lto = true
# opt-level = "s"  # 优化大小
# codegen-units = 1

# 5. wee_alloc：Rust 中使用微型分配器
# Cargo.toml:
# [dependencies]
# wee_alloc = "0.4"
# 在 lib.rs 中:
# #[global_allocator]
# static ALLOC: wee_alloc::WeeAlloc = wee_alloc::WeeAlloc::INIT;
```

## 八、调试和分析工具

```bash
# Chrome DevTools
# - Sources 面板查看 WASM 反汇编
# - Memory 面板分析 WASM 内存
# - Performance 面板分析 WASM 函数耗时

# Firefox Developer Tools
# - 调试器支持 WASM 源码映射
# - Profiler 支持 WASM 函数分析

# wasm-tools（Rust 生态）
cargo install wasm-tools
wasm-tools print module.wasm        # 打印 WAT
wasm-tools strip module.wasm        # 移除自定义段
wasm-tools objdump module.wasm      # 类似 objdump

# twiggy（代码大小分析）
cargo install twiggy
twiggy top module.wasm              # 按大小排序函数
twiggy dominators module.wasm       # 分析支配关系
twiggy paths module.wasm -f "func_name"  # 追溯引用链
```

## 九、社区资源

- **WebAssembly 官方网站**：webassembly.org
- **WASM 规范**：github.com/WebAssembly/spec
- **Bytecode Alliance**：bytecodealliance.org（WASI、Wasmtime）
- **Rust WASM 工作组**：rustwasm.github.io
- **MDN WebAssembly 文档**：developer.mozilla.org/en-US/docs/WebAssembly
