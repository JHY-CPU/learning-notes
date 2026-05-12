# Tokio运行时

## 一、概念说明

Tokio 是 Rust 最流行的异步运行时，提供多线程调度器、I/O 驱动和定时器。

```rust
#[tokio::main]
async fn main() {
    tokio::spawn(async {
        println!("异步任务");
    });
}
```

## 二、具体用法

### 2.1 运行时配置

```rust
#[tokio::main(flavor = "multi_thread", worker_threads = 4)]
async fn main() {
    // 4个工作线程
}

// 手动配置
fn main() {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4)
        .enable_all()
        .build()
        .unwrap();

    rt.block_on(async {
        // 异步代码
    });
}
```

### 2.2 任务管理

```rust
use tokio::task;

async fn example() {
    let handle = task::spawn(async {
        "任务结果"
    });

    let result = handle.await.unwrap();
}
```

### 2.3 IO 操作

```rust
use tokio::fs;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

async fn read_file() -> std::io::Result<String> {
    let mut file = fs::File::open("file.txt").await?;
    let mut contents = String::new();
    file.read_to_string(&mut contents).await?;
    Ok(contents)
}
```

### 2.4 定时器与间隔

```rust
use tokio::time::{interval, sleep, Duration};

async fn timer_examples() {
    // 延迟执行
    sleep(Duration::from_secs(1)).await;
    println!("1秒后执行");

    // 定时间隔
    let mut interval = interval(Duration::from_secs(5));
    loop {
        interval.tick().await;
        println!("每5秒执行一次");
    }

    // 超时处理
    let result = tokio::time::timeout(
        Duration::from_secs(3),
        async {
            sleep(Duration::from_secs(10)).await;
            "完成"
        }
    ).await;

    match result {
        Ok(value) => println!("成功: {}", value),
        Err(_) => println!("超时"),
    }
}
```

### 2.5 信号量控制并发

```rust
use tokio::sync::Semaphore;
use std::sync::Arc;

async fn rate_limited_requests() {
    let semaphore = Arc::new(Semaphore::new(10)); // 最多10个并发
    let mut handles = vec![];

    for i in 0..100 {
        let permit = semaphore.clone().acquire_owned().await.unwrap();
        handles.push(tokio::spawn(async move {
            // 模拟请求
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
            println!("请求 {} 完成", i);
            drop(permit); // 释放许可
        }));
    }

    for handle in handles {
        handle.await.unwrap();
    }
}
```

### 2.6 异步文件操作

```rust
use tokio::fs;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

async fn async_file_ops() -> std::io::Result<()> {
    // 异步写入
    let mut file = fs::File::create("output.txt").await?;
    file.write_all(b"Hello, async!").await?;

    // 异步读取
    let mut file = fs::File::open("input.txt").await?;
    let mut contents = String::new();
    file.read_to_string(&mut contents).await?;

    // 异步读取整个文件
    let contents = fs::read_to_string("data.txt").await?;

    Ok(())
}
```

### 2.7 tokio-console 调试

```rust
// Cargo.toml 添加依赖
// console-subscriber = "0.2"

fn init_console() {
    console_subscriber::init();
}

// 然后用 tokio-console 连接
// $ tokio-console
```

## 四、Tokio 运行时架构

```
Tokio 运行时
├── 多线程调度器
│   ├── 工作线程 1（本地队列）
│   ├── 工作线程 2（本地队列）
│   └── 工作线程 N（work-stealing）
├── I/O 驱动
│   ├── epoll/kqueue/IOCP
│   └── 事件通知
├── 定时器驱动
│   ├── sleep/interval/timeout
│   └── 时间轮
└── 任务队列
    ├── 全局队列
    └── 本地队列（每个线程）
```

## 五、注意事项与常见陷阱

1. **阻塞操作**：使用 `spawn_blocking` 处理阻塞操作，避免阻塞运行时
2. **取消安全**：确保 Future 的取消是安全的，不要在取消时丢失数据
3. **资源泄漏**：注意任务和资源的清理，使用 `JoinHandle` 等待任务完成
4. **背压处理**：使用信号量控制并发度，避免资源耗尽
5. **调试工具**：使用 tokio-console 调试异步任务和性能问题
6. **运行时配置**：根据工作负载配置运行时参数
7. **优雅关闭**：监听信号，等待所有任务完成
