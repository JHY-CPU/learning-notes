# WASM内存模型

## 一、概念说明

WASM 使用线性内存模型，是一个可增长的字节数组。JavaScript 和 WASM 可以共享这块内存。

```javascript
// 创建 WASM 内存
const memory = new WebAssembly.Memory({ initial: 1, maximum: 10 });
// 1 页 = 64KB
```

## 二、具体用法

### 2.1 内存操作

```javascript
// 写入内存
const view = new Uint8Array(memory.buffer);
view[0] = 42;

// 读取内存
console.log(view[0]); // 42

// 内存增长
memory.grow(1); // 增长 1 页 (64KB)
```

### 2.2 WASM 与 JavaScript 共享内存

```javascript
// WASM 模块导入内存
const imports = {
  env: {
    memory: memory,
  }
};

const instance = await WebAssembly.instantiate(bytes, imports);

// JavaScript 读取 WASM 写入的数据
const pointer = instance.exports.getStringPointer();
const length = instance.exports.getStringLength();
const bytes = new Uint8Array(memory.buffer, pointer, length);
const text = new TextDecoder().decode(bytes);
```

### 2.3 内存布局

```javascript
// WASM 线性内存布局
// 0x0000 - 0x0FFF: 栈空间
// 0x1000 - ...: 堆空间
// ... - 最大值: 动态分配

// 不同编译器有不同的内存布局
```

## 三、注意事项与常见陷阱

1. **内存限制**：WASM 内存有最大值限制
2. **字节序**：WASM 使用小端字节序
3. **对齐要求**：访问未对齐数据有性能影响
4. **垃圾回收**：WASM 没有垃圾回收，需要手动管理
5. **内存泄漏**：不当管理会导致内存泄漏

## 四、WAT 中的内存操作

在 WAT 文本格式中直接操作内存：

```wat
(module
  ;; 定义 1 页内存（64KB），最大可增长到 10 页
  (memory 1 10)

  ;; 写入 i32 到内存
  (func $write_i32 (param $addr i32) (param $value i32)
    local.get $addr
    local.get $value
    i32.store)           ;; 内存[$addr] = $value（4 字节）

  ;; 从内存读取 i32
  (func $read_i32 (param $addr i32) (result i32)
    local.get $addr
    i32.load)            ;; return 内存[$addr]

  ;; 写入 i64（8 字节）
  (func $write_i64 (param $addr i32) (param $value i64)
    local.get $addr
    local.get $value
    i64.store)

  ;; 写入单个字节
  (func $write_byte (param $addr i32) (param $value i32)
    local.get $addr
    local.get $value
    i32.store8)          ;; 只写入最低 1 字节

  ;; 数据段：初始化内存
  (data (i32.const 0) "Hello")      ;; 在偏移 0 写入 "Hello"
  (data (i32.const 100) "\01\02\03") ;; 在偏移 100 写入字节序列

  (export "write_i32" (func $write_i32))
  (export "read_i32" (func $read_i32))
  (export "memory" (memory 0))
)
```

## 五、内存布局详解

不同编译工具链的内存布局有所区别：

```
Emscripten 内存布局：
┌──────────────────────────────┐ 0x0000
│        Stack（栈）            │ 向下增长
├──────────────────────────────┤
│        (保留空间)             │
├──────────────────────────────┤ sbrk 基址
│        Heap（堆）             │ 向上增长
├──────────────────────────────┤
│      动态分配区域             │
└──────────────────────────────┘ 最大内存

Rust wasm-bindgen 内存布局：
┌──────────────────────────────┐ 0x0000
│      数据段（静态数据）       │
├──────────────────────────────┤
│        Heap（堆）             │ 使用 dlmalloc/wee_alloc
├──────────────────────────────┤
│        Stack（栈）            │ 向下增长
└──────────────────────────────┘
```

## 六、JavaScript 与 WASM 共享内存的高级用法

### 6.1 使用 TextEncoder/TextDecoder 传递字符串

```javascript
function passStringToWasm(instance, str) {
  const memory = instance.exports.memory;
  const encoder = new TextEncoder();
  const encoded = encoder.encode(str);

  // 在 WASM 中分配空间
  const ptr = instance.exports.alloc(encoded.length);

  // 写入数据
  const memoryView = new Uint8Array(memory.buffer);
  memoryView.set(encoded, ptr);

  return { ptr, length: encoded.length };
}

function getStringFromWasm(instance, ptr, length) {
  const memory = instance.exports.memory;
  const bytes = new Uint8Array(memory.buffer, ptr, length);
  return new TextDecoder().decode(bytes);
}
```

### 6.2 传递数组数据

```javascript
function passArrayToWasm(instance, array) {
  const memory = instance.exports.memory;
  const bytesPerElement = array.BYTES_PER_ELEMENT;
  const byteLength = array.length * bytesPerElement;

  const ptr = instance.exports.alloc(byteLength);
  const memoryView = new Uint8Array(memory.buffer, ptr, byteLength);

  // 将 TypedArray 数据复制到 WASM 内存
  const sourceView = new Uint8Array(array.buffer, array.byteOffset, byteLength);
  memoryView.set(sourceView);

  return { ptr, length: array.length };
}

// 使用示例
const floatArray = new Float64Array([1.0, 2.0, 3.0, 4.0]);
const { ptr, length } = passArrayToWasm(instance, floatArray);
const sum = instance.exports.sum_array(ptr, length);
console.log('数组求和:', sum);
```

### 6.3 SharedArrayBuffer 实现多线程共享

```javascript
// 主线程创建共享内存
const sharedMemory = new WebAssembly.Memory({
  initial: 1,
  maximum: 10,
  shared: true    // 关键：启用共享
});

// 在多个 Worker 中使用同一块内存
const worker1 = new Worker('worker.js');
const worker2 = new Worker('worker.js');

worker1.postMessage({ memory: sharedMemory, id: 1 });
worker2.postMessage({ memory: sharedMemory, id: 2 });

// Worker 线程中
// const memory = event.data.memory;
// const view = new Int32Array(memory.buffer);
// Atomics.add(view, 0, 1);  // 原子操作保证线程安全
```

## 七、内存分配策略

在 WASM 中，常用的内存分配策略包括：

```wat
;; 简单的 bump allocator（线性分配器）
(module
  (memory 1)
  (global $heap_ptr (mut i32) (i32.const 1024))  ;; 从 1024 开始分配

  (func $alloc (param $size i32) (result i32)
    (local $ptr i32)
    global.get $heap_ptr
    local.set $ptr

    ;; 推进指针
    global.get $heap_ptr
    local.get $size
    i32.add
    global.set $heap_ptr

    ;; 返回旧指针
    local.get $ptr)

  ;; 注意：此分配器不支持 free，适合一次性分配场景
  (export "alloc" (func $alloc))
  (export "memory" (memory 0))
)
```

## 八、性能注意事项

- **对齐访问**：使用对齐的 load/store 指令（如 `i32.load` 对齐到 4 字节）性能最优
- **避免频繁 grow**：`memory.grow` 操作开销较大，应预估好初始大小
- **批量操作**：使用 `memory.copy` 和 `memory.fill`（bulk memory 提案）进行大块内存操作
- **TypedArray 视图缓存**：每次 `memory.grow` 后，之前创建的 `ArrayBuffer` 视图会失效，需要重新创建
