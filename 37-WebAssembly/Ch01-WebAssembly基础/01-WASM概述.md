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

## 四、WASM 的发展历史

WebAssembly 的发展经历了多个阶段：

- **2015 年**：Mozilla 发布 ASM.js，为 WASM 奠定了基础
- **2017 年**：四大浏览器厂商（Chrome、Firefox、Safari、Edge）达成共识，发布 WASM MVP
- **2019 年**：WASM 正式成为 W3C 推荐标准
- **2022 年后**：多项扩展提案陆续推进，包括 GC、线程、SIMD、异常处理等

## 五、完整示例：从 WAT 到浏览器

下面展示一个完整的 WASM 模块，包含加法函数和内存操作：

```wat
;; module.wat - 完整示例
(module
  ;; 导入 JavaScript 提供的内存
  (import "env" "memory" (memory 1))

  ;; 导出加法函数
  (func $add (param $a i32) (param $b i32) (result i32)
    local.get $a
    local.get $b
    i32.add)
  (export "add" (func $add))

  ;; 导出阶乘函数
  (func $factorial (param $n i32) (result i32)
    (local $result i32)
    (local $i i32)
    i32.const 1
    local.set $result
    i32.const 1
    local.set $i
    (block $break
      (loop $continue
        local.get $i
        local.get $n
        i32.gt_s
        br_if $break
        local.get $result
        local.get $i
        i32.mul
        local.set $result
        local.get $i
        i32.const 1
        i32.add
        local.set $i
        br $continue))
    local.get $result)
  (export "factorial" (func $factorial))

  ;; 数据段：在内存偏移 0 处写入字符串
  (data (i32.const 0) "Hello, WASM!"))
```

编译并加载到浏览器：

```bash
# 使用 WABT 工具编译
wat2wasm module.wasm module.wat
```

```javascript
// main.js - 浏览器中加载和使用 WASM
const memory = new WebAssembly.Memory({ initial: 1 });

async function loadWasm() {
  const response = await fetch('module.wasm');
  const bytes = await response.arrayBuffer();

  const { instance } = await WebAssembly.instantiate(bytes, {
    env: { memory }
  });

  // 调用导出函数
  console.log('add(2, 3) =', instance.exports.add(2, 3));         // 5
  console.log('factorial(5) =', instance.exports.factorial(5));    // 120

  // 读取内存中的字符串
  const view = new Uint8Array(memory.buffer, 0, 12);
  const text = new TextDecoder().decode(view);
  console.log('内存中的字符串:', text); // "Hello, WASM!"
}

loadWasm();
```

## 六、WASM 的使用场景

WASM 适用于以下典型场景：

| 场景 | 说明 | 示例 |
|------|------|------|
| 图像/视频处理 | 像素级运算密集 | FFmpeg、libvips |
| 游戏引擎 | 实时渲染和物理计算 | Unity、Godot |
| 科学计算 | 大规模数值运算 | TensorFlow.js |
| 加解密 | 高性能密码学操作 | OpenSSL 编译到 WASM |
| 音频处理 | 实时音频效果器 | Web Audio + WASM |
| CAD/3D 建模 | 复杂几何运算 | AutoCAD Web |
| 区块链 | 智能合约执行 | Polkadot、Near |

## 七、WASM 与 JavaScript 的协作模式

WASM 并非要取代 JavaScript，而是作为补充。典型的协作模式：

```javascript
// 模式 1：WASM 处理计算密集部分
// JavaScript 负责 UI 和逻辑控制
class ImageProcessor {
  constructor(wasmInstance) {
    this.wasm = wasmInstance;
  }

  async processImage(imageData) {
    // 将图像数据写入 WASM 内存
    const inputPtr = this.wasm.exports.alloc(imageData.length);
    const memory = new Uint8Array(this.wasm.exports.memory.buffer);
    memory.set(imageData, inputPtr);

    // 调用 WASM 函数处理
    const outputPtr = this.wasm.exports.process_image(inputPtr, imageData.length);

    // 读取结果
    const result = memory.slice(outputPtr, outputPtr + imageData.length);

    // 释放内存
    this.wasm.exports.free(inputPtr);
    this.wasm.exports.free(outputPtr);

    return result;
  }
}

// 模式 2：使用 wasm-bindgen 自动生成绑定（Rust 项目推荐）
import init, { process_image } from './pkg/image_lib.js';

async function main() {
  await init();
  const result = process_image(imageData); // 直接调用，类型安全
}
```

## 八、性能考量

WASM 的性能特征取决于具体场景：

- **CPU 密集型任务**：WASM 通常比 JavaScript 快 2-10 倍，尤其是涉及大量数值运算的场景
- **DOM 操作**：JavaScript 更高效，因为 WASM 调用 DOM 需要通过 JS 胶水代码
- **启动时间**：WASM 二进制格式解析比 JavaScript 快约 20 倍
- **内存效率**：WASM 使用线性内存，没有 GC 停顿，性能更可预测
