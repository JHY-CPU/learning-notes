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

### 2.4 与 Option::take 的对比

```rust
use std::mem::{self, ManuallyDrop};

fn option_take_vs_manual() {
    // Option::take —— 安全但需要 Option 包装
    let mut opt = Some(vec![1, 2, 3]);
    let taken = opt.take(); // opt 变为 None，taken 有所有权
    // 安全：不会忘记清理

    // ManuallyDrop —— 需要 unsafe，但更灵活
    let mut manual = ManuallyDrop::new(vec![4, 5, 6]);
    let data = unsafe { ManuallyDrop::take(&mut manual) };
    // 手动管理，但可以获得原始值的所有权
    println!("{:?}", data);
}
```

### 2.5 资源池中的应用

```rust
use std::mem::ManuallyDrop;

struct ConnectionPool {
    connections: Vec<ManuallyDrop<String>>, // 模拟连接
}

impl ConnectionPool {
    fn new() -> Self {
        ConnectionPool { connections: Vec::new() }
    }

    fn acquire(&mut self) -> Option<String> {
        self.connections.pop().map(|conn| {
            // 从 ManuallyDrop 中取出所有权
            unsafe { ManuallyDrop::into_inner(conn) }
        })
    }

    fn release(&mut self, conn: String) {
        // 放入 ManuallyDrop，池管理生命周期
        self.connections.push(ManuallyDrop::new(conn));
    }
}

impl Drop for ConnectionPool {
    fn drop(&mut self) {
        // 池销毁时手动释放所有连接
        for conn in &mut self.connections {
            unsafe { ManuallyDrop::drop(conn); }
        }
    }
}
```

### 2.6 在 unsafe 代码中的安全模式

```rust
use std::mem::ManuallyDrop;

// 安全模式：用 RAII 包装 ManuallyDrop
struct SafeWrapper<T> {
    inner: ManuallyDrop<T>,
}

impl<T> SafeWrapper<T> {
    fn new(value: T) -> Self {
        SafeWrapper { inner: ManuallyDrop::new(value) }
    }

    fn into_inner(mut self) -> T {
        let value = unsafe { ManuallyDrop::take(&mut self.inner) };
        std::mem::forget(self); // 防止 Drop 再次释放
        value
    }
}

impl<T> Drop for SafeWrapper<T> {
    fn drop(&mut self) {
        unsafe { ManuallyDrop::drop(&mut self.inner); }
    }
}
```

### 2.7 MaybeUninit 替代方案

```rust
use std::mem::MaybeUninit;

fn maybe_uninit_example() {
    // MaybeUninit<T> 表示可能未初始化的值
    // 比 ManuallyDrop 更底层

    let mut uninit = MaybeUninit::<Vec<i32>>::uninit();

    // 初始化
    unsafe {
        uninit.as_mut_ptr().write(vec![1, 2, 3]);
    }

    // 读取（假设已初始化）
    let value = unsafe { uninit.assume_init() };
    println!("{:?}", value);

    // 注意：MaybeUninit 不调用 drop，需要手动管理
}
```

## 四、ManuallyDrop vs 其他方案对比

| 方案 | 安全性 | 灵活性 | 适用场景 |
|------|--------|--------|---------|
| 自动 Drop | 安全 | 低 | 一般情况 |
| ManuallyDrop | unsafe | 高 | FFI、性能优化 |
| Option::take | 安全 | 中 | 可选的资源转移 |
| MaybeUninit | unsafe | 最高 | 未初始化内存 |

## 五、注意事项与常见陷阱

1. **内存泄漏**：忘记手动 drop 会导致内存泄漏，使用 `into_inner` 可避免
2. **unsafe 必要**：ManuallyDrop 的使用几乎都需要 unsafe，确保安全不变量
3. **替代方案**：考虑使用 `Option::take` 或 `std::mem::replace` 代替 ManuallyDrop
4. **顺序问题**：手动 drop 的顺序需正确，依赖关系反向 drop
5. **调试困难**：泄漏问题难以调试，使用 miri 检测未释放的内存
6. **Double Drop**：`into_inner` 后必须 `forget`，防止二次释放
7. **Send/Sync**：ManuallyDrop 的 Send/Sync 特性与内部类型一致
