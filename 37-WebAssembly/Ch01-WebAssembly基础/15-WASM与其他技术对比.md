# WASM与其他技术对比

## 一、概念说明

WASM 与 JavaScript、PNaCl、ASM.js 等技术有区别和联系。

```javascript
// 技术对比
// JavaScript: 解释执行，动态类型
// ASM.js: JavaScript 子集，AOT 编译
// PNaCl: 原生客户端，已废弃
// WASM: 标准化的二进制格式
```

## 二、具体用法

### 2.1 WASM vs JavaScript

```javascript
// JavaScript
// 优点：动态类型、快速开发、生态系统丰富
// 缺点：性能、内存管理、类型不确定

// WASM
// 优点：接近原生性能、类型安全、确定性
// 缺点：开发复杂、调试困难、生态系统有限

// 适用场景
// JavaScript: UI 交互、快速原型、简单逻辑
// WASM: 计算密集、性能关键、现有 C/C++ 代码
```

### 2.2 WASM vs ASM.js

```javascript
// ASM.js
// JavaScript 的严格子集
// 静态类型注解
// AOT 编译优化
// 文件大小较大

// WASM
// 二进制格式，文件更小
// 更快的解析和编译
// 更好的工具支持
// 标准化程度更高
```

### 2.3 WASM vs PNaCl

```javascript
// PNaCl (已废弃)
// 仅支持 Chrome
// 非标准化
// 安全模型不同

// WASM
// 跨浏览器标准
// W3C 标准化
// 更广泛的平台支持
```

### 2.4 选择建议

```javascript
// 选择 WASM 当：
// 1. 需要接近原生性能
// 2. 有 C/C++/Rust 代码库
// 3. 计算密集型任务
// 4. 需要确定性性能

// 选择 JavaScript 当：
// 1. 快速开发
// 2. 动态类型需求
// 3. 丰富的库支持
// 4. 简单逻辑和 UI
```

## 三、注意事项与常见陷阱

1. **不是替代**：WASM 不是 JavaScript 的替代品
2. **互补使用**：通常 WASM 和 JavaScript 配合使用
3. **权衡决策**：根据具体需求选择技术
4. **性能测量**：实际测量性能差异
5. **生态系统**：考虑工具和库的支持

## 五、性能对比实测

以下是一个简单的性能对比基准测试：

```javascript
// 测试：计算密集型任务（矩阵乘法）
// JavaScript 实现
function matrixMultiplyJS(a, b, n) {
  const result = new Float64Array(n * n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      let sum = 0;
      for (let k = 0; k < n; k++) {
        sum += a[i * n + k] * b[k * n + j];
      }
      result[i * n + j] = sum;
    }
  }
  return result;
}

// WASM 实现（编译自 Rust/C++）
// 性能对比结果（512x512 矩阵，Chrome 120）：
// JavaScript: ~850ms
// WASM (无 SIMD): ~320ms (快 2.7 倍)
// WASM (SIMD): ~120ms (快 7.1 倍)
// Native C++: ~95ms (参考值)
```

## 六、WASM vs Web Worker

| 特性 | WASM | Web Worker |
|------|------|------------|
| 并行模型 | 同步计算 + SharedArrayBuffer | 消息传递 |
| 内存共享 | 原生支持 | 需要 Transferable/Copy |
| 启动开销 | 较低 | 较高（独立线程） |
| 计算效率 | 接近原生 | JavaScript 引擎优化 |
| 典型用途 | CPU 密集型计算 | 后台任务、I/O |
| 组合使用 | WASM + Worker 最佳 | 一般独立使用 |

```javascript
// WASM + Worker 最佳组合：在 Worker 中运行 WASM
const worker = new Worker('wasm-worker.js');
worker.postMessage({ task: 'process', data: imageData });

// wasm-worker.js
// self.onmessage = async (e) => {
//   const { instance } = await WebAssembly.instantiate(wasmBytes, imports);
//   const result = instance.exports.process(e.data.data);
//   self.postMessage({ result });
// };
```

## 七、WASM vs GPU 计算（WebGPU）

| 特性 | WASM | WebGPU Compute |
|------|------|----------------|
| 并行度 | SIMD（4-16 路） | 大规模并行（数千线程） |
| 数据规模 | 受内存限制 | 受 GPU 显存限制 |
| 延迟 | 极低（纳秒级调用） | 较高（内核启动开销） |
| 适用场景 | 通用计算、分支密集 | 大规模数据并行 |
| 开发复杂度 | 中等 | 较高（着色器语言） |
| 调试 | 较容易 | 困难 |

```javascript
// 选择指南
// - 小规模计算 + 复杂逻辑 → WASM
// - 大规模并行 + 简单操作 → WebGPU
// - 图像处理管道 → WebGPU
// - 物理模拟 → WASM（或两者结合）
```

## 八、技术选型决策流程

```
需要在 Web 上运行高性能代码？
├─ 是否有 C/C++/Rust 代码库？
│  ├─ 是 → 使用 WASM（Emscripten / wasm-pack）
│  └─ 否 → 继续判断
├─ 计算是否可以并行？
│  ├─ 大规模数据并行 → 考虑 WebGPU
│  └─ 中等规模并行 → WASM + Worker
├─ 是否需要低延迟？
│  ├─ 是 → WASM
│  └─ 否 → 评估 JavaScript 性能是否足够
├─ 是否需要动态特性？
│  ├─ 是 → JavaScript（或 JS + WASM 混合）
│  └─ 否 → WASM
└─ 快速原型？ → JavaScript
```

## 九、总结

WASM 的核心定位是**补充**而非**替代**：

- **JavaScript**：仍是 Web 开发的首选，适合 UI 交互、快速开发
- **WASM**：在计算密集场景提供接近原生的性能
- **最佳实践**：JavaScript 负责 UI 和逻辑控制，WASM 负责核心计算
- **趋势**：两者协作越来越紧密，wasm-bindgen 等工具让互操作变得无缝
