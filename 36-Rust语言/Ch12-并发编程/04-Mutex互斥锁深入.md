# Mutex互斥锁深入

## 一、概念说明

`Mutex<T>` 提供互斥访问，确保同一时刻只有一个线程可以访问数据。Rust 的 Mutex 设计保证了锁的正确使用。

```rust
use std::sync::{Arc, Mutex};
use std::thread;

let counter = Arc::new(Mutex::new(0));
let mut handles = vec![];

for _ in 0..10 {
    let counter = Arc::clone(&counter);
    handles.push(thread::spawn(move || {
        let mut num = counter.lock().unwrap();
        *num += 1;
    }));
}
```

## 二、具体用法

### 2.1 锁的获取

```rust
use std::sync::Mutex;

let m = Mutex::new(5);

// 阻塞获取锁
{
    let mut num = m.lock().unwrap();
    *num = 10;
}

// 非阻塞尝试获取锁
match m.try_lock() {
    Ok(mut num) => *num = 20,
    Err(_) => println!("无法获取锁"),
}
```

### 2.2 锁中毒处理

```rust
use std::sync::Mutex;
use std::thread;

let m = Arc::new(Mutex::new(0));
let m_clone = Arc::clone(&m);

let handle = thread::spawn(move || {
    let mut data = m_clone.lock().unwrap();
    *data = 1;
    panic!("发生错误"); // 锁被中毒
});

handle.join().unwrap_err();

// 尝试获取中毒的锁
match m.lock() {
    Ok(mut data) => *data = 2,
    Err(poisoned) => {
        let data = poisoned.into_inner();
        *data = 3;
    }
}
```

### 2.3 保护复杂状态

```rust
use std::sync::Mutex;
use std::collections::HashMap;

struct State {
    counter: u64,
    data: HashMap<String, Vec<i32>>,
}

let state = Arc::new(Mutex::new(State {
    counter: 0,
    data: HashMap::new(),
}));
```

## 三、注意事项与常见陷阱

1. **死锁避免**：不要嵌套获取同一锁
2. **锁粒度**：锁的持有时间应尽可能短
3. **panic 安全**：持有锁时 panic 会污染锁
4. **替代方案**：简单计数器用 AtomicUsize 更高效
5. **公平性**：std Mutex 不保证公平性
