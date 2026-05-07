# WASM调试技巧

## 一、概念说明

WASM 调试还在发展中，但现代浏览器提供了基本的调试支持。

```javascript
// 使用 Chrome DevTools 调试 WASM
// 1. 打开 DevTools
// 2. Sources 面板
// 3. 找到 WASM 模块
// 4. 设置断点
```

## 二、具体用法

### 2.1 Chrome DevTools

```javascript
// 启用 WASM 调试
// chrome://flags/#enable-webassembly-debugging

// 在 Sources 面板中
// 可以查看 WASM 二进制、反汇编、设置断点

// 配合 Source Maps
// 生成 .wasm.map 文件
```

### 2.2 日志调试

```javascript
// 导入日志函数
const imports = {
  env: {
    log: (value) => console.log('WASM:', value),
    log_string: (ptr, len) => {
      const bytes = new Uint8Array(instance.exports.memory.buffer, ptr, len);
      console.log(new TextDecoder().decode(bytes));
    }
  }
};
```

### 2.3 错误捕获

```javascript
// 捕获 WASM 运行时错误
try {
  instance.exports.problematicFunction();
} catch (error) {
  console.error('WASM 错误:', error);
  console.error('堆栈:', error.stack);
}

// 使用 wasm-bindgen 的错误处理
import init, { fallible_function } from './pkg/my_lib.js';
try {
  await init();
  fallible_function();
} catch (error) {
  console.error('Rust 错误:', error);
}
```

### 2.4 性能分析

```javascript
// 使用 Performance API
performance.mark('wasm-start');
instance.exports.heavyComputation();
performance.mark('wasm-end');
performance.measure('wasm', 'wasm-start', 'wasm-end');

// 使用 Chrome DevTools Performance 面板
// 可以看到 WASM 函数的执行时间
```

## 三、注意事项与常见陷阱

1. **调试信息**：Release 构建可能没有调试信息
2. **Source Maps**：需要配置 Source Maps 支持
3. **优化影响**：优化后的代码可能难以调试
4. **异步调试**：异步 WASM 调试更复杂
5. **工具链支持**：不同工具链的调试支持不同
