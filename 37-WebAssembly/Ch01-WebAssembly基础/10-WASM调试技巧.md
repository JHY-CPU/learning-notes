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

## 五、Emscripten 调试配置

Emscripten 提供详细的调试选项：

```bash
# 生成 DWARF 调试信息（最高详细级别）
emcc main.c -g4 -o main.js

# 启用断言（运行时检查）
emcc main.c -g -s ASSERTIONS=1 -s SAFE_HEAP=1

# SAFE_HEAP：检测内存越界和未对齐访问
emcc main.c -s SAFE_HEAP=1 -s ASSERTIONS=2

# 生成 Source Map（指定 base URL）
emcc main.c -g4 --source-map-base http://localhost:8080/ -o main.js
```

SAFE_HEAP 模式会在运行时检测：
- 读写已释放的内存
- 内存地址未对齐
- 访问超出分配范围的内存

## 六、使用 wasm-objdump 进行离线调试

```bash
# 反汇编 WASM 模块，查看指令级代码
wasm-objdump -d module.wasm

# 查看自定义段（包含调试信息）
wasm-objdump -j name -x module.wasm

# 查看所有段的大小分布
wasm-objdump -h module.wasm

# 输出示例（反汇编）：
# 000045: func[0] <add>
#  000047: 20 00                      | local.get 0
#  000049: 20 01                      | local.get 1
#  00004b: 6a                         | i32.add
#  00004c: 0b                         | end
```

## 七、自定义调试函数

在 WAT 中定义调试辅助函数：

```wat
(module
  (import "env" "debug_i32" (func $debug_i32 (param i32)))
  (import "env" "debug_f64" (func $debug_f64 (param f64)))
  (import "env" "debug_string" (func $debug_string (param i32 i32)))

  (memory 1)

  ;; 调试宏模拟：在关键位置插入调试输出
  (func $complex_calc (param $x i32) (result i32)
    (local $temp i32)

    ;; 打印输入参数
    local.get $x
    call $debug_i32

    ;; 中间计算
    local.get $x
    i32.const 10
    i32.mul
    local.set $temp

    ;; 打印中间结果
    local.get $temp
    call $debug_i32

    local.get $temp
    i32.const 3
    i32.add)

  (export "complex_calc" (func $complex_calc))
)
```

```javascript
// JavaScript 端提供调试函数
const imports = {
  env: {
    debug_i32: (value) => console.log('[WASM i32]', value),
    debug_f64: (value) => console.log('[WASM f64]', value),
    debug_string: (ptr, len) => {
      const bytes = new Uint8Array(memory.buffer, ptr, len);
      console.log('[WASM str]', new TextDecoder().decode(bytes));
    }
  }
};
```

## 八、使用 Console.trace 追踪调用栈

```javascript
// 包装 WASM 导出函数，自动记录调用
function wrapWithTrace(instance) {
  const wrapped = {};
  for (const [name, fn] of Object.entries(instance.exports)) {
    if (typeof fn === 'function') {
      wrapped[name] = (...args) => {
        console.trace(`WASM 调用: ${name}(${args.join(', ')})`);
        const result = fn(...args);
        console.log(`WASM 返回: ${name} = ${result}`);
        return result;
      };
    } else {
      wrapped[name] = fn;
    }
  }
  return wrapped;
}

// 使用
const debug = wrapWithTrace(instance);
debug.add(2, 3); // 会打印调用栈和返回值
```

## 九、常见调试场景

| 场景 | 调试方法 | 工具 |
|------|---------|------|
| 函数返回值错误 | 单步执行、检查栈 | Chrome DevTools |
| 内存越界 | SAFE_HEAP 模式 | Emscripten |
| 导入函数未定义 | 检查导入对象 | wasm-objdump |
| 性能瓶颈 | Performance Profiler | Chrome DevTools |
| 编译失败 | 检查 WAT 语法 | wat2wasm |
| 内存泄漏 | 跟踪分配/释放 | 自定义分配器包装 |
