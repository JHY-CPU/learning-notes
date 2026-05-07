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
