# Emscripten API

## 一、概念说明

Emscripten 提供 JavaScript API 与编译后的 WASM 交互。

```javascript
// Emscripten 生成的 API
var Module = {
  onRuntimeInitialized: function() {
    console.log('运行时初始化完成');
    var result = Module._add(1, 2);
    console.log('结果:', result);
  }
};
```

## 二、具体用法

### 2.1 Module 配置

```javascript
var Module = {
  // 初始化完成回调
  onRuntimeInitialized: function() {
    // WASM 可用
  },

  // 错误处理
  onAbort: function(what) {
    console.error('中止:', what);
  },

  // 打印输出
  print: function(text) {
    console.log('stdout:', text);
  },

  printErr: function(text) {
    console.error('stderr:', text);
  },

  // 画布
  canvas: document.getElementById('canvas'),

  // 内存配置
  INITIAL_MEMORY: 16777216, // 16MB
  TOTAL_MEMORY: 67108864,   // 64MB
};
```

### 2.2 C 函数调用

```javascript
// 直接调用 C 函数
var result = Module._add(1, 2);

// 使用 ccall
var result = Module.ccall(
  'add',        // 函数名
  'number',     // 返回类型
  ['number', 'number'],  // 参数类型
  [1, 2]        // 参数值
);

// 使用 cwrap
var add = Module.cwrap('add', 'number', ['number', 'number']);
var result = add(1, 2);
```

### 2.3 内存操作

```javascript
// 读取内存
var ptr = Module._malloc(100);
var heap = Module.HEAPU8;
var data = heap.slice(ptr, ptr + 100);

// 写入内存
var data = new Uint8Array([1, 2, 3, 4, 5]);
var ptr = Module._malloc(data.length);
Module.HEAPU8.set(data, ptr);

// 释放内存
Module._free(ptr);
```

### 2.4 字符串处理

```javascript
// C 字符串转 JavaScript
var cString = Module._getString();
var jsString = Module.UTF8ToString(cString);

// JavaScript 字符串转 C
var jsString = 'Hello, World!';
var cString = Module.stringToUTF8OnStack(jsString);
Module._printString(cString);
```

## 三、注意事项与常见陷阱

1. **异步加载**：WASM 加载是异步的
2. **内存管理**：手动管理内存分配和释放
3. **类型转换**：正确处理 C 和 JavaScript 类型
4. **性能**：频繁调用有开销
5. **错误处理**：处理 C 代码的错误
