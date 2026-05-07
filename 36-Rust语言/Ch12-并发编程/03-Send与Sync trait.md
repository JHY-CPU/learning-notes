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

## 三、注意事项与常见陷阱

1. **自动实现**：编译器自动推导 Send + Sync
2. **unsafe impl**：手动实现需保证线程安全
3. **Rc 限制**：Rc 不是 Send，不能跨线程
4. **裸指针**：裸指针不是 Send + Sync
5. **生命周期**：引用跨线程需满足生命周期约束
