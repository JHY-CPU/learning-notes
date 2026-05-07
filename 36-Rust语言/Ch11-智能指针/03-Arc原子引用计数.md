# Arc原子引用计数

## 一、概念说明

`Arc<T>`（Atomic Reference Counted）是 `Rc<T>` 的线程安全版本。它使用原子操作进行引用计数，可以在多线程环境中安全地共享数据。

```rust
use std::sync::Arc;
use std::thread;

let data = Arc::new(vec![1, 2, 3, 4, 5]);
let mut handles = vec![];

for _ in 0..3 {
    let data = Arc::clone(&data);
    handles.push(thread::spawn(move || {
        println!("数据: {:?}", data);
    }));
}

for h in handles {
    h.join().unwrap();
}
```

## 二、具体用法

### 2.1 线程间共享

```rust
use std::sync::Arc;
use std::thread;

let counter = Arc::new(0);
let mut handles = vec![];

for _ in 0..10 {
    let counter = Arc::clone(&counter);
    handles.push(thread::spawn(move || {
        // Arc 内部不可变，需要配合 Mutex 修改
        let val = *counter;
        val
    }));
}

let results: Vec<i32> = handles.into_iter()
    .map(|h| h.join().unwrap())
    .collect();
```

### 2.2 Arc + Mutex 可变共享

```rust
use std::sync::{Arc, Mutex};
use std::thread;

let data = Arc::new(Mutex::new(vec![]));
let mut handles = vec![];

for i in 0..5 {
    let data = Arc::clone(&data);
    handles.push(thread::spawn(move || {
        let mut vec = data.lock().unwrap();
        vec.push(i);
    }));
}

for h in handles {
    h.join().unwrap();
}

println!("结果: {:?}", *data.lock().unwrap());
```

### 2.3 Arc + RwLock 读写分离

```rust
use std::sync::{Arc, RwLock};
use std::thread;

let data = Arc::new(RwLock::new(vec![1, 2, 3]));

// 多个读取者
let mut readers = vec![];
for _ in 0..5 {
    let data = Arc::clone(&data);
    readers.push(thread::spawn(move || {
        let vec = data.read().unwrap();
        vec.iter().sum::<i32>()
    }));
}

// 写入者
let data_clone = Arc::clone(&data);
let writer = thread::spawn(move || {
    let mut vec = data_clone.write().unwrap();
    vec.push(4);
});

writer.join().unwrap();
```

## 三、注意事项与常见陷阱

1. **原子开销**：Arc 比 Rc 有额外的原子操作开销
2. **Send + Sync**：Arc 内部类型需实现 Send + Sync
3. **死锁风险**：使用 Mutex/RwLock 时需避免死锁
4. **弱引用**：Arc 也支持 Weak 避免循环引用
5. **克隆成本**：Arc::clone 只增加引用计数，不复制数据
