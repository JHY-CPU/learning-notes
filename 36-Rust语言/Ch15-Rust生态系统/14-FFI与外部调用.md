# FFI与外部调用

## 一、概念说明

Rust 通过 FFI（外部函数接口）与 C、C++ 等语言互操作。

```rust
extern "C" {
    fn abs(input: i32) -> i32;
}

fn main() {
    unsafe {
        println!("abs(-3) = {}", abs(-3));
    }
}
```

## 二、具体用法

### 2.1 调用 C 函数

```rust
// 声明外部函数
extern "C" {
    fn printf(format: *const u8, ...) -> i32;
}

unsafe {
    let msg = b"Hello, FFI!\n\0";
    printf(msg.as_ptr());
}
```

### 2.2 暴露 Rust 函数

```rust
#[no_mangle]
pub extern "C" fn add(a: i32, b: i32) -> i32 {
    a + b
}

// C 头文件
// int32_t add(int32_t a, int32_t b);
```

### 2.3 绑定生成

```rust
// 使用 bindgen 从 C 头文件生成绑定
// build.rs
extern crate bindgen;

fn main() {
    let bindings = bindgen::Builder::default()
        .header("wrapper.h")
        .generate()
        .unwrap();

    bindings.write_to_file("src/bindings.rs").unwrap();
}
```

### 2.4 C 字符串处理

```rust
use std::ffi::{CStr, CString};

unsafe {
    let c_str = CStr::from_ptr(ptr);
    let rust_str = c_str.to_str().unwrap();

    let c_string = CString::new("Hello").unwrap();
    let ptr = c_string.as_ptr();
}
```

### 2.5 repr(C) 与内存布局

```rust
// repr(C) 保证与 C 兼容的内存布局
#[repr(C)]
pub struct CStruct {
    pub x: i32,
    pub y: i32,
    pub data: *mut u8,
    pub len: usize,
}

// repr(u8) 用于枚举
#[repr(u8)]
pub enum Status {
    Ok = 0,
    Error = 1,
    Pending = 2,
}
```

### 2.6 cbindgen 生成头文件

```toml
# Cargo.toml
[lib]
crate-type = ["cdylib", "staticlib"]

[build-dependencies]
cbindgen = "0.26"
```

```rust
// build.rs
fn main() {
    let crate_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    cbindgen::Builder::new()
        .with_crate(crate_dir)
        .generate()
        .expect("Unable to generate bindings")
        .write_to_file("target/include/mylib.h");
}
```

### 2.7 错误码约定

```rust
#[repr(i32)]
pub enum FFIResult {
    Ok = 0,
    InvalidInput = -1,
    InternalError = -2,
    OutOfMemory = -3,
}

#[no_mangle]
pub extern "C" fn process_data(ptr: *const u8, len: usize) -> FFIResult {
    if ptr.is_null() || len == 0 {
        return FFIResult::InvalidInput;
    }

    // 处理数据...

    FFIResult::Ok
}
```

## 四、FFI 安全检查清单

```
Rust 调用 C：
- [ ] extern "C" 声明正确
- [ ] 类型映射正确
- [ ] 内存管理责任明确
- [ ] 空指针检查
- [ ] 线程安全确认

C 调用 Rust：
- [ ] #[no_mangle] 标记
- [ ] extern "C" 约定
- [ ] repr(C) 结构体
- [ ] 内存所有权明确
- [ ] 错误码约定
```

## 五、注意事项与常见陷阱

1. **内存管理**：明确谁负责释放内存，使用 `Box::into_raw` 和 `Box::from_raw` 配对
2. **类型转换**：正确处理 C 和 Rust 类型，注意大小和对齐
3. **错误处理**：C 没有 Result，需要约定错误码或使用输出参数
4. **线程安全**：确保 FFI 调用是线程安全的，标记 Send + Sync
5. **测试覆盖**：为 FFI 边界编写充分测试，包括边界条件
6. **字符串处理**：正确处理 C 字符串（null 结尾），使用 `CString` 和 `CStr`
7. **可移植性**：注意不同平台的类型大小差异，使用 `c_int` 而非 `i32`
