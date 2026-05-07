# Mutex互斥锁

## 一、概念说明

`Mutex<T>` 提供了互斥锁，确保在多线程环境中对共享数据的独占访问。需要先获取锁才能访问内部数据。

```rust
use std::sync::{Arc, Mutex};
use std::thread;

let data = Arc::new(Mutex::new(0));
let mut handles = vec![];

for _ in 0..10 {
    let data = Arc::clone(&data);
    handles.push(thread::spawn(move || {
        let mut num = data.lock().unwrap();
        *num += 1;
    }));
}
```

## 二、具体用法

### 2.1 基本使用

```rust
use std::sync::Mutex;

let m = Mutex::new(5);

{
    let mut num = m.lock().unwrap();
    *num = 6;
} // 锁在此释放

println!("值: {:?}", m);
```

### 2.2 线程间共享

```rust
use std::sync::{Arc, Mutex};
use std::thread;

fn concurrent_increment() -> i32 {
    let counter = Arc::new(Mutex::new(0));
    let mut handles = vec![];

    for _ in 0..10 {
        let counter = Arc::clone(&counter);
        handles.push(thread::spawn(move || {
            for _ in 0..100 {
                let mut num = counter.lock().unwrap();
                *num += 1;
            }
        }));
    }

    for h in handles {
        h.join().unwrap();
    }

    *counter.lock().unwrap()
}
```

### 2.3 错误处理

```rust
use std::sync::Mutex;

let m = Mutex::new(5);

match m.lock() {
    Ok(mut num) => {
        *num += 1;
    }
    Err(poisoned) => {
        // 锁被污染（持有锁的线程 panic）
        let num = poisoned.into_inner();
        println!("恢复数据: {:?}", num);
    }
}
```

## 三、注意事项与常见陷阱

1. **死锁**：同一锁嵌套获取会导致死锁
2. **锁粒度**：锁的持有时间应尽可能短
3. **panic 传播**：持有锁时 panic 会污染锁
4. **性能**：竞争激烈时考虑分片或无锁算法
5. **单线程**：std::sync::Mutex 是线程安全的，单线程使用 std::sync::Mutex
