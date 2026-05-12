# Send与Sync trait

## 一、概念说明

`Send` 和 `Sync` 是 Rust 并发安全的基石。`Send` 表示类型可以安全地在线程间转移所有权，`Sync` 表示类型可以安全地在线程间共享引用。

```rust
// Send: T 可以安全发送到另一个线程
// Sync: &T 可以安全地在多个线程间共享

// 几乎所有类型都自动实现 Send 和 Sync
// Rc<T> 不是 Send，因为引用计数不是原子的
// Arc<T> 是 Send，因为使用原子引用计数
```

## 二、具体用法

### 2.1 自动派生

```rust
// 大多数类型自动实现 Send + Sync
// Copy 类型都是 Send + Sync
// 包含 Send 字段的结构体自动实现 Send

struct MyData {
    value: i32,
    name: String,
}
// MyData 自动实现 Send

// 包含非 Send 类型则不是 Send
use std::rc::Rc;
struct NotSend {
    data: Rc<i32>,
}
// NotSend 不实现 Send
```

### 2.2 手动实现

```rust
// 标记非 Send 类型为 Send（需 unsafe）
struct CustomPtr(*mut i32);

// 不安全！CustomPtr 不自动实现 Send
// 但可以手动实现（如果能保证安全）
unsafe impl Send for CustomPtr {}

// 同步原语自动实现 Send + Sync
use std::sync::{Mutex, Arc};
// Arc<Mutex<T>> 是 Send + Sync
```

### 2.3 trait bound 约束

```rust
use std::thread;

fn spawn_with_data<F, T>(f: F) -> thread::JoinHandle<T>
where
    F: FnOnce() -> T + Send + 'static,
    T: Send + 'static,
{
    thread::spawn(f)
}

// 只有满足 Send 约束的闭包才能跨线程
let data = String::from("hello");
spawn_with_data(move || {
    println!("{}", data);
    data.len()
});
```

### 2.4 Send + Sync 规则详解

```rust
// Send 规则：
// - 所有基本类型（i32, bool 等）是 Send
// - 包含 Send 字段的结构体是 Send
// - Rc<T> 不是 Send（非原子引用计数）
// - *mut T / *const T 不是 Send（裸指针）
// - Cell<T> 不是 Sync，但可以是 Send（如果 T 是 Send）

// Sync 规则：
// - 如果 &T 是 Send，则 T 是 Sync
// - 所有基本类型是 Sync
// - Mutex<T> 是 Sync（即使 T 不是 Sync）
// - Cell<T> 不是 Sync（内部可变性）

fn send_sync_examples() {
    use std::sync::{Arc, Mutex};
    use std::rc::Rc;
    use std::cell::Cell;

    // Send: 可以跨线程转移所有权
    let data = Arc::new(Mutex::new(42));
    std::thread::spawn(move || {
        *data.lock().unwrap() += 1;
    });

    // Rc 不是 Send，不能跨线程
    // let rc = Rc::new(42);
    // std::thread::spawn(move || { rc; }); // 编译错误

    // Cell 不是 Sync，但可以是 Send
    let cell = Cell::new(42);
    std::thread::spawn(move || {
        cell.set(100);
    }); // Cell 是 Send，可以 move 到新线程
}
```

### 2.5 自定义类型的 Send/Sync

```rust
use std::sync::{Arc, Mutex};

// 包含 Send 字段的结构体自动实现 Send
struct MySendType {
    value: i32,
    name: String,
}
// MySendType 自动实现 Send

// 包含非 Send 字段则不是 Send
use std::rc::Rc;
struct NotSendType {
    rc_data: Rc<i32>,
}
// NotSendType 不实现 Send

// 手动实现 Send（需要 unsafe）
struct RawPtrWrapper(*mut i32);
unsafe impl Send for RawPtrWrapper {}

// 需要保证：
// 1. 数据可以安全地跨线程传递
// 2. 没有数据竞争
// 3. 满足 Rust 的并发安全要求
```

### 2.6 使用 trait bound 编写通用并发代码

```rust
use std::sync::{Arc, Mutex};
use std::thread;

fn concurrent_map<F, T, U>(data: &[T], f: F) -> Vec<U>
where
    F: Fn(&T) -> U + Send + Sync,
    T: Send + Sync,
    U: Send,
{
    let result = Arc::new(Mutex::new(Vec::with_capacity(data.len())));

    let handles: Vec<_> = data.chunks(data.len() / 4 + 1)
        .map(|chunk| {
            let result = Arc::clone(&result);
            let f = &f;
            thread::spawn(move || {
                let mapped: Vec<U> = chunk.iter().map(f).collect();
                result.lock().unwrap().extend(mapped);
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    Arc::try_unwrap(result).unwrap().into_inner().unwrap()
}
```

### 2.7 PhantomData 与标记类型

```rust
use std::marker::PhantomData;

// 使用 PhantomData 控制 Send/Sync 特性
struct MyType<T> {
    data: *const u8,
    _marker: PhantomData<T>,
}

// 如果 T 是 Send，则 MyType<T> 是 Send
unsafe impl<T: Send> Send for MyType<T> {}

// 如果 T 是 Sync，则 MyType<T> 是 Sync
unsafe impl<T: Sync> Sync for MyType<T> {}
```

## 四、并发安全保证的层次

```
编译器检查
├── 所有权系统 → 防止数据竞争
├── 借用规则 → 防止悬垂引用
├── 生命周期 → 确保引用有效
├── Send trait → 允许跨线程转移所有权
└── Sync trait → 允许跨线程共享引用

运行时保证
├── Mutex → 互斥访问
├── RwLock → 读写分离
├── Atomic → 原子操作
└── Channel → 消息传递
```

## 五、注意事项与常见陷阱

1. **自动实现**：编译器自动推导 Send + Sync，但手动实现需要 unsafe
2. **unsafe impl**：手动实现必须保证线程安全，违反会导致未定义行为
3. **Rc 限制**：Rc 不是 Send，不能跨线程，多线程使用 Arc
4. **裸指针**：裸指针不是 Send + Sync，需要手动标记
5. **生命周期**：引用跨线程需满足生命周期约束，通常需要 `'static`
6. **Cell/RefCell**：Cell 不是 Sync，RefCell 也不是，需要线程安全使用 Mutex
7. **PhantomData**：可以用 PhantomData 传播 Send/Sync 特性到自定义类型
