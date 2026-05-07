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

## 三、注意事项与常见陷阱

1. **阻塞操作**：使用 spawn_blocking 处理阻塞
2. **取消安全**：确保 Future 的取消是安全的
3. **资源泄漏**：注意任务和资源的清理
4. **背压处理**：使用信号量控制并发度
5. **调试工具**：使用 tokio-console 调试
