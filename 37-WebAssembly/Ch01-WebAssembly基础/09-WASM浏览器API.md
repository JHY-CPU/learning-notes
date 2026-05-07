# WASM浏览器API

## 一、概念说明

浏览器提供了 JavaScript API 来加载、编译和实例化 WASM 模块。

```javascript
// 核心 API
// WebAssembly.compile() - 编译 WASM
// WebAssembly.instantiate() - 实例化 WASM
// WebAssembly.Module - 编译后的模块
// WebAssembly.Instance - 实例化后的实例
// WebAssembly.Memory - 线性内存
// WebAssembly.Table - 函数表
```

## 二、具体用法

### 2.1 加载和编译

```javascript
// 方式 1: compile + instantiate
const response = await fetch('module.wasm');
const bytes = await response.arrayBuffer();
const module = await WebAssembly.compile(bytes);
const instance = await WebAssembly.instantiate(module, imports);

// 方式 2: instantiateStreaming（推荐）
const { instance } = await WebAssembly.instantiateStreaming(
  fetch('module.wasm'),
  imports
);

// 方式 3: compileStreaming
const module = await WebAssembly.compileStreaming(fetch('module.wasm'));
const instance = await WebAssembly.instantiate(module, imports);
```

### 2.2 使用实例

```javascript
// 调用导出函数
const result = instance.exports.add(1, 2);

// 访问导出内存
const memory = instance.exports.memory;
const view = new Uint8Array(memory.buffer);

// 访问导出全局变量
const global = instance.exports.myGlobal;
console.log(global.value);
```

### 2.3 错误处理

```javascript
try {
  const { instance } = await WebAssembly.instantiateStreaming(
    fetch('module.wasm'),
    imports
  );
} catch (error) {
  if (error instanceof WebAssembly.CompileError) {
    console.error('编译错误:', error);
  } else if (error instanceof WebAssembly.LinkError) {
    console.error('链接错误:', error);
  } else if (error instanceof WebAssembly.RuntimeError) {
    console.error('运行时错误:', error);
  }
}
```

### 2.4 流式加载

```javascript
// 流式加载支持渐进式编译
const response = fetch('large-module.wasm');
const { instance } = await WebAssembly.instantiateStreaming(response, imports);

// 配合 Service Worker 缓存
self.addEventListener('fetch', (event) => {
  event.respondWith(
    caches.match(event.request).then((response) => {
      return response || fetch(event.request);
    })
  );
});
```

## 三、注意事项与常见陷阱

1. **MIME 类型**：服务器需要正确设置 `application/wasm`
2. **CORS**：WASM 文件需要正确的 CORS 头
3. **缓存策略**：利用浏览器缓存 WASM 文件
4. **内存管理**：注意 WASM 内存的生命周期
5. **旧浏览器**：需要检测 WASM 支持并提供降级方案
