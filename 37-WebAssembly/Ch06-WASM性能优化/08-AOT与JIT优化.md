# AOT与JIT优化

## 一、概念说明

WASM 默认使用 AOT（提前编译）模式，V8 引擎在某些场景也使用 JIT（即时编译）。理解编译策略有助于优化。

```
WASM 字节码 → Liftoff (基线编译) → TurboFan (优化编译)
                ↓ 快速启动             ↓ 高性能
```

## 二、具体用法

### 2.1 V8 编译策略

```javascript
// V8 内部编译流程：
// 1. Liftoff：基线编译器，快速生成代码
// 2. TurboFan：优化编译器，生成高效代码

// 通过 V8 标志观察（仅调试用）
// --trace-wasm-compilation-times
// --trace-wasm-decoder
```

### 2.2 代码布局优化

```rust
// 热点函数放在前面，提高编译优先级
#[inline(always)]
pub fn hot_path(input: &[u8]) -> u32 {
    // 高频执行的代码
    input.iter().map(|&b| b as u32).sum()
}

// 冷代码放在后面
#[cold]
fn cold_error_handling(err: &str) -> ! {
    panic!("{}", err);
}
```

### 2.3 分支预测优化

```rust
// 使用 likely/unlikely 提示编译器
#[cold]
fn unlikely_branch() -> bool { false }

pub fn optimized_match(value: u32) -> u32 {
    if value < 100 {
        // 常见路径（编译器会优化）
        value * 2
    } else if value < 1000 {
        value * 3
    } else {
        // 罕见路径
        value.saturating_mul(10)
    }
}
```

### 2.4 编译时间提示

```rust
// 内联策略
#[inline]        // 编译器自行决定
#[inline(always)] // 总是内联（谨慎使用）
#[inline(never)]  // 从不内联

// 小函数建议内联
#[inline]
fn clamp(value: f32, min: f32, max: f32) -> f32 {
    value.max(min).min(max)
}

// 大函数不建议内联
#[inline(never)]
fn complex_calculation(data: &[u8]) -> Vec<u8> {
    // 复杂逻辑...
    data.to_vec()
}
```

### 2.5 编译配置文件

```toml
# 针对不同场景的优化
[profile.release]
opt-level = "s"      # 体积优先
lto = true
codegen-units = 1

[profile.release-fast]
inherits = "release"
opt-level = 3         # 速度优先
lto = "thin"          # 更快的 LTO

[profile.release-debug]
inherits = "release"
debug = true          # 保留调试信息
strip = false
```

```bash
# 构建时选择 profile
cargo build --profile release-fast --target wasm32-unknown-unknown
```

### 2.6 预热策略

```javascript
// 预热 WASM 代码，触发 JIT 优化
async function warmupWasm(instance) {
    // 多次调用热点函数，触发 TurboFan 优化
    const testInput = new Uint8Array(1024);
    for (let i = 0; i < 100; i++) {
        instance.exports.process(testInput.byteOffset, testInput.length);
    }
    console.log('WASM 预热完成');
}
```

## 三、注意事项与常见陷阱

1. **AOT vs JIT**：WASM 主要是 AOT，JIT 优化有限
2. **编译延迟**：首次加载需编译，大型模块启动慢
3. **代码缓存**：浏览器可能缓存编译结果，但不可依赖
4. **优化副作用**：过度内联可能增大代码体积
5. **平台差异**：不同浏览器引擎的优化策略不同
