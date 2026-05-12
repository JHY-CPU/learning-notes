# WASM性能特征

## 一、概念说明

WASM 设计为高性能，接近原生速度。主要性能优势来自编译型语言和优化的二进制格式。

```javascript
// WASM 性能特点
// 1. 编译型：预先编译为机器码
// 2. 二进制格式：快速解析和编译
// 3. 无垃圾回收：确定性性能
// 4. SIMD 支持：并行计算
```

## 二、具体用法

### 2.1 性能基准

```javascript
// JavaScript vs WASM 性能比较
const jsResult = fibonacci(40); // JavaScript
const wasmResult = instance.exports.fibonacci(40); // WASM

// CPU 密集型任务 WASM 通常快 2-10 倍
// I/O 密集型任务差异不大
```

### 2.2 优化建议

```javascript
// 1. 减少 JavaScript-WASM 边界调用
// 不好：频繁调用
for (let i = 0; i < 1000; i++) {
  instance.exports.process(data[i]);
}

// 好：批量调用
instance.exports.processBatch(data, 1000);

// 2. 使用 SharedArrayBuffer
const sharedBuffer = new SharedArrayBuffer(1024);
const memory = new WebAssembly.Memory({ initial: 1, shared: true });
```

### 2.3 内存优化

```javascript
// 内存池模式
(module
  (memory 1)
  (global $heap_ptr (mut i32) (i32.const 0))

  (func $malloc (param $size i32) (result i32)
    (local $ptr i32)
    global.get $heap_ptr
    local.set $ptr
    global.get $heap_ptr
    local.get $size
    i32.add
    global.set $heap_ptr
    local.get $ptr))
```

## 三、注意事项与常见陷阱

1. **加载时间**：WASM 编译有时间开销
2. **内存开销**：WASM 线性内存可能较大
3. **JavaScript 优化**：JavaScript 引擎已经很优化
4. **实际收益**：性能收益取决于具体场景
5. **Profile**：使用开发者工具分析性能

## 四、WASM 执行流水线

理解 WASM 的执行流水线有助于优化加载性能：

```
获取二进制 → 解析 → 验证 → 编译 → 实例化 → 执行
   ↓          ↓       ↓       ↓        ↓       ↓
  网络I/O   极快    类型检查  JIT/AOT  分配内存  运行代码
  (瓶颈)   (~20x   (~1ms)  (~10ms)  (~1ms)  (最快)
          快于JS)
```

优化加载时间的策略：

```javascript
// 1. 使用流式编译（推荐）：边下载边编译
const { instance } = await WebAssembly.instantiateStreaming(
  fetch('module.wasm'),
  imports
);

// 2. 预编译并缓存模块
const module = await WebAssembly.compileStreaming(fetch('module.wasm'));
// 将 module 传递给多个 Worker，无需重复编译
const instance1 = await WebAssembly.instantiate(module, imports);

// 3. 使用 IndexedDB 缓存编译结果
async function cacheCompiledModule(url) {
  const cache = await caches.open('wasm-cache');
  const response = await fetch(url);
  const module = await WebAssembly.compileStreaming(response.clone());
  // 存储原始字节（编译后的 Module 对象无法直接序列化）
  await cache.put(url, response);
  return module;
}
```

## 五、SIMD 性能加速

SIMD（Single Instruction Multiple Data）可以显著提升数据并行计算的性能：

```wat
;; SIMD 向量点积 vs 标量实现的对比
(module
  (memory 1)

  ;; 标量版本：逐个元素计算
  (func $dot_product_scalar (param $a i32) (param $b i32) (param $len i32) (result f32)
    (local $i i32)
    (local $sum f32)
    (local.set $sum (f32.const 0))
    (block $break
      (loop $continue
        (br_if $break (i32.ge_u (local.get $i) (local.get $len)))
        (local.set $sum
          (f32.add
            (local.get $sum)
            (f32.mul
              (f32.load (i32.add (local.get $a) (i32.shl (local.get $i) (i32.const 2))))
              (f32.load (i32.add (local.get $b) (i32.shl (local.get $i) (i32.const 2)))))))
        (local.set $i (i32.add (local.get $i) (i32.const 1)))
        (br $continue)))
    (local.get $sum))

  ;; SIMD 版本：一次处理 4 个 f32
  ;; 性能提升约 2-4 倍
  (func $dot_product_simd (param $a i32) (param $b i32) (param $len i32) (result f32)
    (local $i i32)
    (local $sum v128)
    (local.set $sum (f32x4.splat (f32.const 0)))
    (block $break
      (loop $continue
        (br_if $break (i32.ge_u (local.get $i) (local.get $len)))
        (local.set $sum
          (f32x4.add
            (local.get $sum)
            (f32x4.mul
              (v128.load (i32.add (local.get $a) (i32.shl (local.get $i) (i32.const 2))))
              (v128.load (i32.add (local.get $b) (i32.shl (local.get $i) (i32.const 2)))))))
        (local.set $i (i32.add (local.get $i) (i32.const 16)))  ;; 每次前进 16 字节 (4个f32)
        (br $continue)))
    ;; 水平求和
    (f32.add
      (f32.add (f32x4.extract_lane 0 (local.get $sum)) (f32x4.extract_lane 1 (local.get $sum)))
      (f32.add (f32x4.extract_lane 2 (local.get $sum)) (f32x4.extract_lane 3 (local.get $sum)))))

  (export "dot_product_scalar" (func $dot_product_scalar))
  (export "dot_product_simd" (func $dot_product_simd))
)
```

## 六、多线程性能

使用 SharedArrayBuffer 和 Atomics 实现多线程：

```javascript
// 主线程
const sharedMemory = new WebAssembly.Memory({
  initial: 10,    // 640KB
  maximum: 100,
  shared: true
});

const numWorkers = 4;
const workers = [];
for (let i = 0; i < numWorkers; i++) {
  const worker = new Worker('compute-worker.js');
  worker.postMessage({
    memory: sharedMemory,
    workerId: i,
    numWorkers: numWorkers,
    taskSize: 1000000
  });
  workers.push(worker);
}

// Worker 线程中
// self.onmessage = async (e) => {
//   const { memory, workerId, numWorkers, taskSize } = e.data;
//   const chunkSize = taskSize / numWorkers;
//   const start = workerId * chunkSize;
//   const end = start + chunkSize;
//
//   // 加载并实例化 WASM 模块
//   const { instance } = await WebAssembly.instantiate(
//     wasmBytes, { env: { memory } }
//   );
//
//   // 每个 Worker 处理数据的不同部分
//   instance.exports.process_chunk(start, end);
//
//   // 使用 Atomics 通知主线程
//   Atomics.add(new Int32Array(memory.buffer), 0, 1);
// };
```

## 七、性能基准测试

```javascript
// 使用 performance.now() 进行精确基准测试
function benchmark(name, fn, iterations = 1000) {
  // 预热
  for (let i = 0; i < 10; i++) fn();

  const start = performance.now();
  for (let i = 0; i < iterations; i++) fn();
  const end = performance.now();

  const avgMs = (end - start) / iterations;
  console.log(`${name}: 平均 ${avgMs.toFixed(4)}ms, 总计 ${(end - start).toFixed(2)}ms`);
  return avgMs;
}

// 对比 JavaScript 和 WASM
benchmark('JavaScript fibonacci(35)', () => jsFibonacci(35));
benchmark('WASM fibonacci(35)', () => instance.exports.fibonacci(35));

// 典型结果：
// JavaScript fibonacci(35): 平均 120.5ms
// WASM fibonacci(35): 平均 25.3ms
// WASM 约快 4.8 倍
```

## 八、何时不应使用 WASM

并非所有场景都适合 WASM：

- **简单的 DOM 操作**：JavaScript 更直接，WASM 需要通过 JS 桥接
- **I/O 密集型任务**：网络、文件读写等，性能瓶颈不在计算
- **快速原型开发**：JavaScript/TypeScript 开发效率更高
- **小规模计算**：WASM 调用开销可能超过计算收益
- **需要动态特性的场景**：如动态类型、运行时代码生成
