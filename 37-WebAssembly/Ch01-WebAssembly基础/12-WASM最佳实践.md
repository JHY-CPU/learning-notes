# WASM最佳实践

## 一、概念说明

总结 WASM 开发的最佳实践，帮助编写高效、可维护的 WASM 代码。

```javascript
// 最佳实践清单
// 1. 选择合适的源语言
// 2. 最小化 JavaScript-WASM 边界调用
// 3. 使用批量操作
// 4. 优化内存使用
// 5. 利用流式编译
```

## 二、具体用法

### 2.1 性能优化

```javascript
// 批量处理数据
const processData = (data) => {
  // 不好：逐个处理
  for (let i = 0; i < data.length; i++) {
    instance.exports.process(data[i]);
  }

  // 好：批量处理
  const offset = instance.exports.alloc(data.length);
  const memory = new Uint8Array(instance.exports.memory.buffer);
  memory.set(data, offset);
  instance.exports.processBatch(offset, data.length);
};
```

### 2.2 内存管理

```javascript
// 使用内存池
class WasmMemoryPool {
  constructor(instance) {
    this.instance = instance;
    this.allocated = new Set();
  }

  alloc(size) {
    const ptr = this.instance.exports.malloc(size);
    this.allocated.add(ptr);
    return ptr;
  }

  free(ptr) {
    this.instance.exports.free(ptr);
    this.allocated.delete(ptr);
  }

  cleanup() {
    for (const ptr of this.allocated) {
      this.instance.exports.free(ptr);
    }
    this.allocated.clear();
  }
}
```

### 2.3 错误处理

```javascript
// 统一错误处理
class WasmError extends Error {
  constructor(message, wasmError) {
    super(message);
    this.wasmError = wasmError;
  }
}

const safeCall = (fn, ...args) => {
  try {
    return fn(...args);
  } catch (error) {
    throw new WasmError('WASM 调用失败', error);
  }
};
```

## 三、注意事项与常见陷阱

1. **过度优化**：不要过早优化
2. **实际测试**：在目标环境测试性能
3. **包大小**：监控 WASM 包大小
4. **兼容性**：测试不同浏览器和设备
5. **维护性**：代码可读性很重要

## 四、包大小优化

```bash
# Rust 项目的大小优化配置
# Cargo.toml
[profile.release]
opt-level = "s"      # 优化大小
lto = true           # 链接时优化
codegen-units = 1    # 单代码生成单元
panic = "abort"      # 使用 abort 而非 unwind
strip = true         # 移除符号表

# 构建后进一步优化
wasm-opt -Os -o output.wasm input.wasm
wasm-strip output.wasm

# 使用 wee_alloc 替代默认分配器（减少 ~10KB）
# #[global_allocator]
# static ALLOC: wee_alloc::WeeAlloc = wee_alloc::WeeAlloc::INIT;

# 移除未使用的依赖和功能
# 使用 cargo tree 检查依赖树
# 使用 feature flags 禁用不需要的功能
```

## 五、异步操作模式

```rust
// 使用 wasm-bindgen-futures 处理异步
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;
use web_sys::{Request, RequestInit, Response, Window};

#[wasm_bindgen]
pub async fn fetch_data(url: &str) -> Result<String, JsValue> {
    let mut opts = RequestInit::new();
    opts.method("GET");

    let request = Request::new_with_str_and_init(url, &opts)?;
    let window = web_sys::window().unwrap();
    let resp_value = JsFuture::from(window.fetch_with_request(&request)).await?;
    let resp: Response = resp_value.dyn_into()?;

    let text = JsFuture::from(resp.text()?).await?;
    Ok(text.as_string().unwrap())
}
```

```javascript
// JavaScript 端使用
import init, { fetch_data } from './pkg/my_lib.js';

await init();
const data = await fetch_data('https://api.example.com/data');
console.log(data);
```

## 六、渐进增强策略

```javascript
// 浏览器兼容性检测和降级方案
async function loadImageProcessor() {
  if (typeof WebAssembly === 'object') {
    try {
      const { processImage } = await import('./wasm-processor.js');
      return processImage;
    } catch (e) {
      console.warn('WASM 加载失败，使用 JavaScript 回退');
    }
  }
  // JavaScript 回退实现
  return (imageData) => jsProcessImage(imageData);
}

const processor = await loadImageProcessor();
processor(imageData); // 自动选择 WASM 或 JS 实现
```

## 七、安全最佳实践

```javascript
// 1. 验证 WASM 模块来源
async function loadTrustedModule(url) {
  // 使用 SRI（Subresource Integrity）验证
  const response = await fetch(url);
  const bytes = await response.arrayBuffer();

  // 验证模块结构
  const module = await WebAssembly.compile(bytes);
  const imports = WebAssembly.Module.imports(module);

  // 检查是否只导入预期的函数
  const allowed = ['env.memory', 'env.log'];
  for (const imp of imports) {
    if (!allowed.includes(`${imp.module}.${imp.name}`)) {
      throw new Error(`未授权的导入: ${imp.module}.${imp.name}`);
    }
  }

  return module;
}

// 2. 限制资源使用
const memory = new WebAssembly.Memory({
  initial: 1,
  maximum: 16  // 最大 1MB
});
```

## 八、文档和 API 设计

```rust
// 为 WASM 导出函数编写清晰的文档
use wasm_bindgen::prelude::*;

/// 计算图像的灰度值
///
/// # 参数
/// - `data`: RGBA 像素数据的指针
/// - `width`: 图像宽度
/// - `height`: 图像高度
///
/// # 返回值
/// 返回灰度图像数据的指针，调用者负责释放
///
/// # Safety
/// 指针必须指向有效的内存区域
#[wasm_bindgen]
pub fn to_grayscale(data: *const u8, width: u32, height: u32) -> *const u8 {
    // 实现...
    std::ptr::null()
}
```

## 九、调试和监控

```javascript
// 性能监控封装
class WasmMonitor {
  constructor(instance) {
    this.instance = instance;
    this.metrics = {};
  }

  wrapExport(name, fn) {
    return (...args) => {
      const start = performance.now();
      const result = fn(...args);
      const elapsed = performance.now() - start;

      if (!this.metrics[name]) {
        this.metrics[name] = { calls: 0, totalTime: 0 };
      }
      this.metrics[name].calls++;
      this.metrics[name].totalTime += elapsed;

      return result;
    };
  }

  report() {
    console.table(this.metrics);
  }
}

// 使用
const monitor = new WasmMonitor(instance);
const add = monitor.wrapExport('add', instance.exports.add);
add(1, 2);
monitor.report(); // 输出性能统计
```
