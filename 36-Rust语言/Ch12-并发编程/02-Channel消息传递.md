# Channel消息传递

## 一、概念说明

Rust 通过 `mpsc`（multiple producer, single consumer）channel 实现线程间消息传递。这是 Go 语言风格的 CSP（通信顺序进程）模式在 Rust 中的体现。

```rust
use std::sync::mpsc;
use std::thread;

let (tx, rx) = mpsc::channel();

thread::spawn(move || {
    tx.send("你好").unwrap();
});

let msg = rx.recv().unwrap();
println!("收到: {}", msg);
```

## 二、具体用法

### 2.1 基本 channel

```rust
use std::sync::mpsc;
use std::thread;

let (tx, rx) = mpsc::channel();

// 发送端在新线程
thread::spawn(move || {
    let messages = vec!["消息1", "消息2", "消息3"];
    for msg in messages {
        tx.send(msg).unwrap();
        thread::sleep(std::time::Duration::from_millis(100));
    }
});

// 接收端在主线程
for received in rx {
    println!("收到: {}", received);
}
```

### 2.2 多生产者

```rust
use std::sync::mpsc;
use std::thread;

let (tx, rx) = mpsc::channel();
let mut handles = vec![];

for i in 0..3 {
    let tx = tx.clone();
    handles.push(thread::spawn(move || {
        tx.send(format!("线程{}的消息", i)).unwrap();
    }));
}

drop(tx); // 关闭原始发送端

for handle in handles {
    handle.join().unwrap();
}

for msg in rx {
    println!("{}", msg);
}
```

### 2.3 同步 channel

```rust
use std::sync::mpsc;
use std::thread;

// sync_channel 有缓冲区限制
let (tx, rx) = mpsc::sync_channel(1); // 缓冲区大小1

thread::spawn(move || {
    tx.send(1).unwrap(); // 立即返回
    tx.send(2).unwrap(); // 阻塞直到接收方消费
});

thread::sleep(std::time::Duration::from_millis(100));
println!("{}", rx.recv().unwrap());
```

## 三、注意事项与常见陷阱

1. **所有权转移**：send 会转移值的所有权
2. **channel 关闭**：所有发送端 drop 后，接收端收到 None
3. **阻塞行为**：recv 会阻塞，try_recv 不会
4. **背压处理**：sync_channel 可以实现背压机制
5. **错误处理**：send/recv 返回 Result，需处理失败情况
