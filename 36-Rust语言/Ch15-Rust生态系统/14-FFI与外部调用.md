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

## 三、注意事项与常见陷阱

1. **内存管理**：明确谁负责释放内存
2. **类型转换**：正确处理 C 和 Rust 类型
3. **错误处理**：C 没有 Result，需要约定错误码
4. **线程安全**：确保 FFI 调用是线程安全的
5. **测试覆盖**：为 FFI 边界编写充分测试
