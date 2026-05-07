# WebAssembly概述

## 一、概念说明

WebAssembly（简称 WASM）是一种低级的二进制指令格式，设计为高级语言（如 C/C++/Rust）的编译目标。它在浏览器中以接近原生速度运行。

```javascript
// JavaScript 加载 WASM 模块
WebAssembly.instantiateStreaming(fetch('module.wasm'))
  .then(({ instance }) => {
    console.log(instance.exports.add(2, 3)); // 5
  });
```

## 二、具体用法

### 2.1 WASM 核心特性

```javascript
// 1. 二进制格式：紧凑、快速解析
// 2. 沙箱环境：安全隔离
// 3. 确定性执行：相同输入产生相同输出
// 4. 内存安全：线性内存模型
// 5. 跨平台：任何支持 WASM 的环境
```

### 2.2 支持的语言

```bash
# C/C++ 使用 Emscripten
emcc hello.c -o hello.wasm

# Rust 使用 wasm-pack
wasm-pack build --target web

# AssemblyScript（TypeScript 子集）
asc hello.ts -o hello.wasm

# Go 使用 TinyGo
tinygo build -o hello.wasm -target wasm ./hello.go
```

### 2.3 浏览器支持

```javascript
// 检测 WASM 支持
if (typeof WebAssembly === 'object') {
  console.log('浏览器支持 WASM');
}

// 主流浏览器都已支持 WASM
// Chrome 57+, Firefox 52+, Safari 11+, Edge 16+
```

### 2.4 基本执行流程

```javascript
// 1. 获取 WASM 二进制
const response = await fetch('module.wasm');
const bytes = await response.arrayBuffer();

// 2. 编译
const module = await WebAssembly.compile(bytes);

// 3. 实例化
const instance = await WebAssembly.instantiate(module, imports);

// 4. 调用导出函数
const result = instance.exports.add(1, 2);
```

## 三、注意事项与常见陷阱

1. **二进制大小**：WASM 文件可能较大，需要优化
2. **调试困难**：WASM 调试工具还在完善中
3. **JavaScript 互操作**：频繁跨边界调用有性能开销
4. **内存管理**：需要手动管理线性内存
5. **浏览器兼容**：虽然主流支持，但老浏览器不支持
