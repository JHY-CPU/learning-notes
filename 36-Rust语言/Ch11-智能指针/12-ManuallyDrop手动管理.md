# ManuallyDrop手动管理

## 一、概念说明

`ManuallyDrop<T>` 抑制编译器自动调用 drop，允许程序员手动控制何时释放资源。用于 FFI、性能优化等需要精细控制的场景。

```rust
use std::mem::ManuallyDrop;

let mut data = ManuallyDrop::new(vec![1, 2, 3]);
// data 不会自动释放

// 手动释放（需要 unsafe）
unsafe { ManuallyDrop::drop(&mut data); }
```

## 二、具体用法

### 2.1 延迟释放

```rust
use std::mem::ManuallyDrop;

fn delayed_drop() {
    let mut data = ManuallyDrop::new(vec![1, 2, 3]);

    // 使用数据
    println!("{:?}", *data);

    // 在合适的时机释放
    unsafe { ManuallyDrop::drop(&mut data); }
}
```

### 2.2 FFI 场景

```rust
use std::mem::ManuallyDrop;

#[repr(C)]
struct FFIStruct {
    data: *mut u8,
    len: usize,
}

fn to_ffi(data: Vec<u8>) -> FFIStruct {
    let mut data = ManuallyDrop::new(data);
    FFIStruct {
        data: data.as_mut_ptr(),
        len: data.len(),
    }
}

fn from_ffi(ffi: FFIStruct) -> Vec<u8> {
    unsafe { Vec::from_raw_parts(ffi.data, ffi.len, ffi.len) }
}
```

### 2.3 部分初始化

```rust
use std::mem::ManuallyDrop;

union MaybeInit<T> {
    uninit: (),
    value: ManuallyDrop<T>,
}

fn maybe_init() {
    let mut u = MaybeInit { uninit: () };
    // 初始化一个字段
    unsafe { u.value = ManuallyDrop::new(42); }
    // 使用
    unsafe { println!("{}", *u.value); }
    // 手动释放
    unsafe { ManuallyDrop::drop(&mut u.value); }
}
```

## 三、注意事项与常见陷阱

1. **内存泄漏**：忘记手动 drop 会导致内存泄漏
2. **unsafe 必要**：ManuallyDrop 的使用几乎都需要 unsafe
3. **替代方案**：考虑使用 Option 和 take 代替 ManuallyDrop
4. **顺序问题**：手动 drop 的顺序需正确
5. **调试困难**：泄漏问题难以调试
