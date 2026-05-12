# Rust 异步编程

## async/await 基础

Rust 的 async/await 是基于状态机的零成本抽象。`async fn` 返回一个实现了 `Future` trait 的类型，只有在被 `.await` 时才会真正执行。

```rust
use tokio; // Cargo.toml: tokio = { version = "1", features = ["full"] }

async fn fetch_data(url: &str) -> String {
    // 模拟异步 IO
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    format!("来自 {} 的数据", url)
}

async fn process(url: &str) -> String {
    let data = fetch_data(url).await; // 等待异步操作完成
    format!("已处理: {}", data)
}

#[tokio::main]
async fn main() {
    let result = process("https://example.com").await;
    println!("{}", result);
}
```

### Future 的本质：状态机编译

Rust 的 async/await 在编译期被转换为状态机，这是 Rust 异步模型的核心。

```rust
// 用户写的代码
async fn read_file(path: &str) -> String {
    let content = fs::read_to_string(path).await;
    content.to_uppercase()
}

// 编译器生成的伪代码（简化）
enum ReadFileStateMachine {
    // 状态 0: 初始状态
    State0 { path: &str },
    // 状态 1: 等待 read_to_string 完成
    State1 { future: ReadToStringFuture },
    // 状态 2: 完成
    Done,
}

impl Future for ReadFileStateMachine {
    type Output = String;

    fn poll(self: Pin<&mut Self>, cx: &mut Context) -> Poll<String> {
        loop {
            match self {
                State0 { path } => {
                    let fut = fs::read_to_string(path);
                    *self = State1 { future: fut };
                    // 继续循环，立即 poll 新的 future
                }
                State1 { future } => {
                    match Pin::new(future).poll(cx) {
                        Poll::Ready(content) => {
                            *self = Done;
                            return Poll::Ready(content.to_uppercase());
                        }
                        Poll::Pending => return Poll::Pending,
                    }
                }
                Done => panic!("polled after completion"),
            }
        }
    }
}
```

```
状态机的内存布局:

普通函数调用栈:
  ┌──────────────────┐
  │ read_file 栈帧    │  ← 函数返回时释放
  │   local vars     │
  ├──────────────────┤
  │ caller 栈帧      │
  └──────────────────┘

async 函数（状态机）:
  ┌──────────────────────────────────────┐
  │ ReadFileStateMachine (堆上或栈上)     │
  │   ┌──────────────────────────────┐   │
  │   │ State0: { path: &str }       │   │ 最大状态所需空间
  │   │ 或                            │   │ = 所有 await 点之间
  │   │ State1: { future: ... }      │   │   局部变量的最大并集
  │   └──────────────────────────────┘   │
  └──────────────────────────────────────┘

关键特性:
  - 状态机大小 = 所有状态中最大那个的大小
  - 每个 await 点是一个状态转换
  - 零运行时开销: 没有堆分配、没有虚函数调用
```

### Pin 和 Unpin

```rust
// Pin 的作用: 保证 Future 不会被移动
// 因为状态机可能包含自引用（如 &str 指向自己的 String 字段）

use std::pin::Pin;
use std::future::Future;

// Future trait 的签名
pub trait Future {
    type Output;
    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output>;
}

// Pin<&mut T> 的保证:
//   T 在内存中的位置不会改变
//   这对于自引用结构是必需的

// Unpin trait: 标记类型可以安全地从 Pin 中移出
// 大多数类型是 Unpin 的
// async fn 生成的 Future 通常不是 Unpin 的
```

## Tokio 运行时架构

`tokio` 是 Rust 生态中最流行的异步运行时，提供多线程调度器、IO 驱动和定时器。

### Tokio 的内部架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                      Tokio 运行时                                    │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ Scheduler (调度器)                                          │   │
│  │                                                             │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │   │
│  │  │ Worker 0    │  │ Worker 1    │  │ Worker N    │        │   │
│  │  │ 本地队列     │  │ 本地队列     │  │ 本地队列     │        │   │
│  │  │ [F1][F2][F3]│  │ [F4][F5]    │  │ [F6][F7][F8]│        │   │
│  │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘        │   │
│  │         │                │                │                │   │
│  │         │  工作窃取       │                │                │   │
│  │         └────────────────┴────────────────┘                │   │
│  │                                                             │   │
│  │  全局注入队列: [F9, F10, F11, ...]                          │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ I/O Driver (IO 驱动)                                       │   │
│  │                                                             │   │
│  │  ┌───────────────┐  ┌───────────────┐                      │   │
│  │  │ epoll/kqueue  │  │ 定时器堆       │                      │   │
│  │  │ (注册的 fd)    │  │ (deadline)    │                      │   │
│  │  └───────────────┘  └───────────────┘                      │   │
│  │                                                             │   │
│  │  I/O 资源注册: TcpStream, UdpSocket, File 等               │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ Blocking Pool (阻塞线程池)                                  │   │
│  │ 用于执行 spawn_blocking 任务（文件 IO、DNS 解析等）          │   │
│  │ 动态扩缩容（最大 512 线程）                                  │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

### Reactor 模式详解

Tokio 使用 Reactor 模式处理 IO 事件：

```
                  应用程序
                     │
         ┌───────────┼───────────┐
         │           │           │
     .await      .await      .await
         │           │           │
         ▼           ▼           ▼
    ┌────────────────────────────────┐
    │        Waker 机制              │
    │  Future 返回 Pending           │
    │  注册 Waker 到 IO 资源         │
    └────────────┬───────────────────┘
                 │
                 ▼
    ┌────────────────────────────────┐
    │        Reactor                 │
    │  epoll_wait() / kevent()       │
    │                                │
    │  IO 事件就绪时:                │
    │  1. 找到对应的 Waker           │
    │  2. 将 Future 重新入队         │
    │  3. Worker 继续 poll 它        │
    └────────────────────────────────┘
```

```rust
// Waker 的底层工作原理（简化）
impl Future for MyTcpRead {
    type Output = io::Result<Vec<u8>>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<Vec<u8>>> {
        // 尝试非阻塞读取
        match self.socket.try_read(&mut self.buf) {
            Ok(n) => Poll::Ready(Ok(self.buf[..n].to_vec())),
            Err(ref e) if e.kind() == io::ErrorKind::WouldBlock => {
                // 数据还没准备好
                // 注册 Waker: 当 epoll 检测到这个 socket 可读时，唤醒这个 Future
                self.socket.register_waker(cx.waker());
                Poll::Pending
            }
            Err(e) => Poll::Ready(Err(e)),
        }
    }
}
```

### 运行时配置

```rust
use tokio::runtime::Builder;

fn main() {
    // 多线程运行时（默认）
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        println!("多线程运行时");
    });

    // 单线程运行时
    let rt = Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    rt.block_on(async {
        println!("当前线程运行时");
    });

    // 自定义线程数
    let rt = Builder::new_multi_thread()
        .worker_threads(4)
        .thread_name("my-worker")
        .enable_all()
        .build()
        .unwrap();
    rt.block_on(async {
        println!("自定义运行时: 4 个工作线程");
    });
}
```

### spawn 异步任务

```rust
#[tokio::main]
async fn main() {
    // spawn 一个新任务
    let handle = tokio::spawn(async {
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        42
    });

    let result = handle.await.unwrap();
    println!("任务结果: {}", result);

    // spawn_blocking: 在阻塞线程池中运行同步代码
    let blocking_handle = tokio::spawn_blocking(|| {
        // 模拟阻塞 IO
        std::thread::sleep(std::time::Duration::from_millis(100));
        "阻塞操作完成"
    });

    println!("{}", blocking_handle.await.unwrap());
}
```

### spawn 的内部机制

```rust
// tokio::spawn 的内部实现（简化）
pub fn spawn<F>(future: F) -> JoinHandle<F::Output>
where
    F: Future + Send + 'static,  // 必须 Send + 'static
    F::Output: Send + 'static,
{
    // 1. 将 Future 包装为 Task（包含状态、Waker 等）
    let task = Task::new(future);

    // 2. 将 Task 放入调度队列
    //    - 如果当前在 worker 线程: 放入本地队列
    //    - 否则: 放入全局注入队列
    scheduler::schedule(task);

    // 3. 返回 JoinHandle
    JoinHandle { ... }
}
```

**为什么需要 `Send + 'static`**：
- `Send`：Future 可能被移动到另一个线程执行
- `'static`：Future 不能持有非静态引用（因为可能活得比调用者更久）

### spawn_local 与 !Send 的 Future

```rust
use tokio::task;

#[tokio::main(flavor = "current_thread")]  // 必须是单线程运行时
async fn main() {
    // spawn_local: 不要求 Send，只能在当前线程执行
    let non_send_data = std::rc::Rc::new(42);  // Rc 不是 Send

    task::spawn_local(async move {
        println!("非 Send 数据: {}", non_send_data);
    });
}
```

## select! 宏

`select!` 同时等待多个异步操作，第一个完成的分支被执行。未完成的分支会被取消。

```rust
use tokio::time::{sleep, Duration};

async fn slow_task() -> &'static str {
    sleep(Duration::from_secs(2)).await;
    "慢任务完成"
}

async fn fast_task() -> &'static str {
    sleep(Duration::from_millis(100)).await;
    "快任务完成"
}

#[tokio::main]
async fn main() {
    // 等待第一个完成的异步操作
    tokio::select! {
        result = slow_task() => {
            println!("慢任务: {}", result);
        }
        result = fast_task() => {
            println!("快任务: {}", result);
        }
    }
}
```

### select! 的内部实现

```
select! 宏展开为类似这样的代码:

// 同时 poll 所有分支
let mut f1 = slow_task();
let mut f2 = fast_task();

loop {
    // 尝试 poll 第一个分支
    if let Poll::Ready(v) = Pin::new(&mut f1).poll(cx) {
        // 取消其他分支
        drop(f2);
        return Branch1(v);
    }
    // 尝试 poll 第二个分支
    if let Poll::Ready(v) = Pin::new(&mut f2).poll(cx) {
        drop(f1);
        return Branch2(v);
    }
    // 都返回 Pending，注册所有分支的 Waker，然后返回 Pending
    return Poll::Pending;
}
```

### 超时与取消

```rust
use tokio::time::{sleep, Duration, timeout};

#[tokio::main]
async fn main() {
    // select! 实现超时
    let result = tokio::select! {
        val = async {
            sleep(Duration::from_secs(5)).await;
            "完成"
        } => format!("正常完成: {}", val),
        _ = sleep(Duration::from_secs(1)) => "超时!".to_string(),
    };
    println!("{}", result);

    // timeout 函数
    match timeout(Duration::from_millis(100), sleep(Duration::from_secs(1))).await {
        Ok(_) => println!("完成了"),
        Err(_) => println!("超时"),
    }
}
```

### select! 循环

```rust
use tokio::sync::mpsc;

#[tokio::main]
async fn main() {
    let (tx, mut rx) = mpsc::channel::<String>(10);

    tokio::spawn(async move {
        for i in 0..5 {
            tx.send(format!("消息-{}", i)).await.unwrap();
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        }
    });

    loop {
        tokio::select! {
            msg = rx.recv() => {
                match msg {
                    Some(data) => println!("收到: {}", data),
                    None => {
                        println!("channel 已关闭");
                        break;
                    }
                }
            }
            _ = tokio::time::sleep(std::time::Duration::from_secs(1)) => {
                println!("接收超时");
                break;
            }
        }
    }
}
```

### select! 的公平性

```rust
// select! 默认行为: 随机选择就绪的分支（类似 Go 的 select）
// biased select!: 按代码顺序选择（更可预测）

tokio::select! {
    biased;  // 使用 biased 模式

    // 优先检查取消信号
    _ = cancel_rx.recv() => {
        println!("被取消");
    }
    // 其次处理数据
    msg = data_rx.recv() => {
        println!("收到数据: {:?}", msg);
    }
    // 最后处理超时
    _ = sleep(Duration::from_secs(5)) => {
        println!("超时");
    }
}
```

## join! 宏

`join!` 同时等待多个异步操作全部完成。

```rust
use tokio::join;

async fn fetch_user(id: u32) -> String {
    tokio::time::sleep(std::time::Duration::from_millis(100 * id as u64)).await;
    format!("User-{}", id)
}

async fn fetch_posts(user_id: u32) -> Vec<String> {
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    vec![format!("Post-{}-1", user_id), format!("Post-{}-2", user_id)]
}

#[tokio::main]
async fn main() {
    // 同时执行多个异步操作
    let (user, posts) = join!(
        fetch_user(1),
        fetch_posts(1)
    );

    println!("用户: {}", user);
    println!("帖子: {:?}", posts);

    // join! 返回元组
    let results = join!(
        fetch_user(1),
        fetch_user(2),
        fetch_user(3)
    );
    println!("结果: {:?}", results);
}
```

### join! vs select!

```
宏        行为                      返回值         适用场景
──────────────────────────────────────────────────────────────────
join!     等待所有完成               元组          并行获取多个数据
select!   等待第一个完成             单个值        超时、取消、竞态
try_join! 等待所有完成，任一失败即停  Result       并行获取，需错误处理
```

```rust
use anyhow::Result;

async fn fetch_a() -> Result<String> {
    Ok("A".to_string())
}

async fn fetch_b() -> Result<String> {
    anyhow::bail!("B 失败了");
}

#[tokio::main]
async fn main() -> Result<()> {
    // try_join!: 任何一个失败立即返回错误
    let (a, b) = tokio::try_join!(fetch_a(), fetch_b())?;
    println!("{} {}", a, b);
    Ok(())
}
```

## 完整工程级示例：异步 HTTP 代理服务器

```rust
use anyhow::Result;
use std::sync::Arc;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::Semaphore;

struct ProxyConfig {
    max_connections: usize,
    target_addr: String,
}

struct ProxyServer {
    config: Arc<ProxyConfig>,
    semaphore: Arc<Semaphore>,
    stats: Arc<tokio::sync::Mutex<ProxyStats>>,
}

struct ProxyStats {
    total_connections: u64,
    active_connections: u64,
    bytes_transferred: u64,
}

impl ProxyServer {
    fn new(config: ProxyConfig) -> Self {
        let max_conn = config.max_connections;
        Self {
            config: Arc::new(config),
            semaphore: Arc::new(Semaphore::new(max_conn)),
            stats: Arc::new(tokio::sync::Mutex::new(ProxyStats {
                total_connections: 0,
                active_connections: 0,
                bytes_transferred: 0,
            })),
        }
    }

    async fn run(&self, addr: &str) -> Result<()> {
        let listener = TcpListener::bind(addr).await?;
        println!("代理服务器启动在 {}", addr);

        loop {
            let (client, peer) = listener.accept().await?;
            println!("新连接来自: {}", peer);

            let permit = self.semaphore.clone().acquire_owned().await?;
            let config = Arc::clone(&self.config);
            let stats = Arc::clone(&self.stats);

            {
                let mut s = stats.lock().await;
                s.total_connections += 1;
                s.active_connections += 1;
            }

            tokio::spawn(async move {
                if let Err(e) = Self::handle_connection(client, &config).await {
                    eprintln!("连接错误: {}", e);
                }
                drop(permit);  // 释放信号量
                let mut s = stats.lock().await;
                s.active_connections -= 1;
            });
        }
    }

    async fn handle_connection(
        mut client: TcpStream,
        config: &ProxyConfig,
    ) -> Result<()> {
        // 连接目标服务器
        let mut upstream = TcpStream::connect(&config.target_addr).await?;

        // 双向转发数据
        let (mut client_read, mut client_write) = client.split();
        let (mut upstream_read, mut upstream_write) = upstream.split();

        let client_to_upstream = async {
            let mut buf = vec![0u8; 8192];
            loop {
                let n = client_read.read(&mut buf).await?;
                if n == 0 { break; }
                upstream_write.write_all(&buf[..n]).await?;
            }
            upstream_write.shutdown().await?;
            Ok::<_, anyhow::Error>(())
        };

        let upstream_to_client = async {
            let mut buf = vec![0u8; 8192];
            loop {
                let n = upstream_read.read(&mut buf).await?;
                if n == 0 { break; }
                client_write.write_all(&buf[..n]).await?;
            }
            client_write.shutdown().await?;
            Ok::<_, anyhow::Error>(())
        };

        // 并发双向转发
        tokio::try_join!(client_to_upstream, upstream_to_client)?;
        Ok(())
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let config = ProxyConfig {
        max_connections: 1000,
        target_addr: "127.0.0.1:8080".to_string(),
    };

    let server = ProxyServer::new(config);
    server.run("0.0.0.0:3000").await?;
    Ok(())
}
```

## tokio::sync 同步原语

### Mutex

```rust
use tokio::sync::Mutex;
use std::sync::Arc;

#[tokio::main]
async fn main() {
    let data = Arc::new(Mutex::new(0));

    let mut handles = vec![];
    for i in 0..10 {
        let data = Arc::clone(&data);
        handles.push(tokio::spawn(async move {
            let mut lock = data.lock().await;
            *lock += 1;
            println!("任务 {}: 计数 = {}", i, *lock);
        }));
    }

    for h in handles {
        h.await.unwrap();
    }

    println!("最终: {}", *data.lock().await);
}
```

### tokio::sync::Mutex vs std::sync::Mutex

```
类型                await 安全?    跨 await 持有?    性能
────────────────────────────────────────────────────────────
std::sync::Mutex    否            是 (但阻塞)       更快 (~17ns)
tokio::sync::Mutex  是            是                略慢 (~50ns)

使用建议:
  - 如果锁持有期间没有 await → 用 std::sync::Mutex
  - 如果需要跨 await 持有锁 → 用 tokio::sync::Mutex
  - 尽量缩小锁的范围

// 正确: 不要在持有 std::Mutex 时 await
let data = std::sync::Mutex::new(0);
{
    let mut val = data.lock().unwrap();
    *val += 1;
} // 释放锁
some_async_function().await;  // OK

// 错误: 持有 std::Mutex 时 await
let val = data.lock().unwrap();
some_async_function().await;  // 可能导致死锁或 panic!
```

### oneshot / mpsc / broadcast channel

```rust
use tokio::sync::{oneshot, mpsc, broadcast};

#[tokio::main]
async fn main() {
    // oneshot: 一次性通信
    let (tx, rx) = oneshot::channel::<String>();
    tokio::spawn(async move {
        tx.send("一次性消息".to_string()).unwrap();
    });
    println!("oneshot: {}", rx.await.unwrap());

    // mpsc: 多生产者单消费者
    let (tx, mut rx) = mpsc::channel::<u32>(32);
    for i in 0..5 {
        let tx = tx.clone();
        tokio::spawn(async move {
            tx.send(i).await.unwrap();
        });
    }
    drop(tx);
    while let Some(val) = rx.recv().await {
        println!("mpsc 收到: {}", val);
    }

    // broadcast: 多生产者多消费者
    let (tx, _rx) = broadcast::channel::<String>(16);
    let mut rx1 = tx.subscribe();
    let mut rx2 = tx.subscribe();

    tokio::spawn(async move {
        tx.send("广播消息".to_string()).unwrap();
    });

    println!("rx1: {}", rx1.recv().await.unwrap());
    println!("rx2: {}", rx2.recv().await.unwrap());
}
```

### Tokio Channel 类型选择

```
Channel 类型    生产者    消费者    缓冲    适用场景
──────────────────────────────────────────────────────────────
oneshot         1         1         0      请求-响应模式、一次性结果
mpsc            多        1         N      任务队列、事件流
broadcast       多        多        N      事件通知、发布-订阅
watch           多        多        1      配置更新、状态广播
```

```rust
// watch channel: 单值广播
use tokio::sync::watch;

#[tokio::main]
async fn main() {
    let (tx, mut rx1) = watch::channel("initial".to_string());
    let mut rx2 = tx.subscribe();

    tokio::spawn(async move {
        // 每次 send 覆盖旧值，所有接收者都能看到最新值
        tx.send("updated".to_string()).unwrap();
    });

    // rx.wait_for 等待值变为指定的条件
    rx1.wait_for(|val| val == "updated").await.unwrap();
    println!("rx1 看到: {}", *rx1.borrow());

    rx2.wait_for(|val| val == "updated").await.unwrap();
    println!("rx2 看到: {}", *rx2.borrow());
}
```

## Async Streams

```rust
use tokio_stream::StreamExt;

async fn number_stream() -> impl tokio_stream::Stream<Item = i32> {
    tokio_stream::iter(vec![1, 2, 3, 4, 5])
}

#[tokio::main]
async fn main() {
    // 使用 StreamExt 处理流
    let mut stream = number_stream();

    while let Some(item) = stream.next().await {
        println!("收到: {}", item);
    }

    // 异步迭代
    use futures::stream;
    let mut s = stream::iter(1..=5)
        .map(|x| x * x)
        .filter(|x| *x > 10);

    while let Some(item) = s.next().await {
        println!("过滤结果: {}", item);
    }
}
```

### async_stream 宏：创建自定义异步流

```rust
use async_stream::stream;
use tokio_stream::StreamExt;

fn countdown(start: u32) -> impl tokio_stream::Stream<Item = u32> {
    stream! {
        let mut count = start;
        while count > 0 {
            yield count;
            count -= 1;
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        }
    }
}

#[tokio::main]
async fn main() {
    let mut s = countdown(5);
    while let Some(n) = s.next().await {
        println!("倒计时: {}", n);
    }
}
```

### 实战：异步事件流处理

```rust
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::StreamExt;

#[derive(Debug)]
enum Event {
    UserLogin { user_id: u64 },
    UserLogout { user_id: u64 },
    Message { from: u64, content: String },
}

async fn process_events(mut stream: impl tokio_stream::Stream<Item = Event> + Unpin) {
    let mut online_users = std::collections::HashSet::new();

    while let Some(event) = stream.next().await {
        match event {
            Event::UserLogin { user_id } => {
                online_users.insert(user_id);
                println!("用户 {} 上线，当前在线: {}", user_id, online_users.len());
            }
            Event::UserLogout { user_id } => {
                online_users.remove(&user_id);
                println!("用户 {} 下线，当前在线: {}", user_id, online_users.len());
            }
            Event::Message { from, content } => {
                println!("用户 {}: {}", from, content);
            }
        }
    }
}

#[tokio::main]
async fn main() {
    let (tx, rx) = mpsc::channel(100);

    tokio::spawn(async move {
        tx.send(Event::UserLogin { user_id: 1 }).await.unwrap();
        tx.send(Event::Message { from: 1, content: "你好!".into() }).await.unwrap();
        tx.send(Event::UserLogout { user_id: 1 }).await.unwrap();
    });

    let stream = ReceiverStream::new(rx);
    process_events(stream).await;
}
```

## 异步文件 IO

```rust
use tokio::fs;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 异步写入文件
    let mut file = fs::File::create("test.txt").await?;
    file.write_all(b"Hello, async Rust!").await?;

    // 异步读取文件
    let mut file = fs::File::open("test.txt").await?;
    let mut contents = String::new();
    file.read_to_string(&mut contents).await?;
    println!("文件内容: {}", contents);

    // 简洁的读写方式
    fs::write("test2.txt", "直接写入").await?;
    let data = fs::read_to_string("test2.txt").await?;
    println!("读取: {}", data);

    // 清理
    fs::remove_file("test.txt").await.ok();
    fs::remove_file("test2.txt").await.ok();

    Ok(())
}
```

### Tokio 文件 IO 的注意事项

```
重要: tokio::fs 的文件操作实际在阻塞线程池中执行!

原因: Linux 上没有真正的异步文件 IO (aio 在 Linux 上不成熟)
      文件 IO 会触发内核预读，无法像网络 IO 那样非阻塞

tokio::fs::read_to_string() 的实际流程:
  1. tokio::spawn_blocking(|| std::fs::read_to_string(path))
  2. 在阻塞线程池中执行同步文件操作
  3. 完成后通过 channel 将结果传回

对于大量文件 IO:
  - 直接使用 std::fs + spawn_blocking 更高效
  - 避免为每个小文件创建单独的异步任务
  - 考虑批量处理
```

```rust
// 高效的批量文件读取
use tokio::task;

async fn read_files(paths: Vec<String>) -> Vec<String> {
    // 将整个批量操作作为一个阻塞任务
    task::spawn_blocking(move || {
        paths.iter()
            .filter_map(|p| std::fs::read_to_string(p).ok())
            .collect()
    }).await.unwrap()
}
```

## 错误处理

```rust
use anyhow::{Result, Context};

async fn fetch_and_process(url: &str) -> Result<String> {
    let data = fetch_url(url).await
        .context(format!("获取 {} 失败", url))?;
    Ok(format!("处理: {}", data))
}

async fn fetch_url(url: &str) -> Result<String> {
    if url.is_empty() {
        anyhow::bail!("URL 不能为空");
    }
    tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    Ok(format!("{} 的内容", url))
}

#[tokio::main]
async fn main() {
    match fetch_and_process("https://example.com").await {
        Ok(result) => println!("{}", result),
        Err(e) => println!("错误: {}", e),
    }
}
```

### thiserror 用于库的错误定义

```rust
use thiserror::Error;

#[derive(Error, Debug)]
enum AppError {
    #[error("网络错误: {0}")]
    Network(#[from] reqwest::Error),

    #[error("IO 错误: {0}")]
    Io(#[from] std::io::Error),

    #[error("数据库错误: {source}")]
    Database {
        source: sqlx::Error,
        query: String,
    },

    #[error("超时: {0} 秒")]
    Timeout(u64),
}

async fn fetch_data(url: &str) -> Result<String, AppError> {
    let response = reqwest::get(url).await?;  // 自动转换为 AppError::Network
    let body = response.text().await?;
    Ok(body)
}
```

## 调试方法论

### 1. 使用 tokio-console 实时监控

```bash
# 安装 tokio-console
cargo install tokio-console

# 在项目中启用 console subscriber
# Cargo.toml:
# [dependencies]
# console-subscriber = "0.2"

# 代码中启用
use tracing_subscriber::prelude::*;

fn main() {
    console_subscriber::init();

    // 或者更精细的配置
    tracing_subscriber::registry()
        .with(console_subscriber::spawn())
        .init();

    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(async { your_app().await });
}

# 运行后在另一个终端:
tokio-console
# 可以看到: 所有 task 的状态、poll 次数、总耗时、waker 来源
```

### 2. 使用 tracing 进行结构化日志

```rust
use tracing::{info, instrument, warn};

#[instrument]  // 自动记录函数调用和参数
async fn fetch_user(id: u64) -> String {
    info!(user_id = id, "开始获取用户");
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    if id == 0 {
        warn!("无效的用户 ID");
    }
    format!("User-{}", id)
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();
    let user = fetch_user(42).await;
    println!("{}", user);
}

// 输出:
// 2024-01-01T00:00:00Z INFO fetch_user{user_id=42}: 开始获取用户
```

### 3. 检测未 poll 的 Future（泄漏检测）

```rust
// 在 debug 模式下，tokio 会警告未 poll 的 Future
// 启用 tokio 的 unhandled_panic 配置

#[tokio::main]
async fn main() {
    // 泄漏检测: 检查未完成的任务
    let handle = tokio::spawn(async {
        tokio::time::sleep(std::time::Duration::from_secs(9999)).await;
    });

    // 不 await handle，任务泄漏
    // tokio 在 runtime drop 时会打印警告

    // 手动检测
    println!("当前任务数: {}", tokio::runtime::Handle::current()
        .metrics().num_alive_tasks());
}
```

### 4. 使用 Miri 检测 unsafe 代码中的 UB

```bash
# Miri 可以检测异步代码中的未定义行为
# 注意: Miri 运行速度较慢，适合测试小片段
rustup +nightly component add miri
cargo +nightly miri test

# 检测特定测试
cargo +nightly miri test test_async_safe
```

### 5. 异步死锁检测

```rust
// 常见异步死锁模式

// 1. 持有 std::sync::Mutex 时 await
// let guard = std_mutex.lock().unwrap();
// some_async().await;  // 死锁风险

// 2. 自己等待自己
// let handle = tokio::spawn(async { ... });
// handle.await  // 如果在同一个 task 中等待，且没有其他 task 运行

// 3. channel 两端都在等待
// let (tx, rx) = mpsc::channel(1);
// tx.send(1).await;  // 如果缓冲区满了
// rx.recv().await;   // 而接收在另一个没人 poll 的 task 中

// 检测方法:
// - 使用 tokio-console 查看阻塞的 task
// - 设置超时: tokio::time::timeout(duration, operation).await
// - 使用 tracing 记录关键路径
```

## 性能实测数据

```
操作                                    耗时 (Tokio 1.x)      说明
──────────────────────────────────────────────────────────────────────
tokio::spawn                            ~0.5-1 μs             创建异步任务
tokio::spawn_blocking                   ~5-10 μs              阻塞线程池任务
channel send (mpsc, 无竞争)             ~30 ns                有缓冲 channel
channel recv (mpsc, 无竞争)             ~25 ns
Mutex::lock (tokio, 无竞争)             ~50 ns                异步 Mutex
Mutex::lock (std, 无竞争)               ~17 ns                同步 Mutex
select! (2 分支)                        ~100 ns               包含 poll 开销
TcpStream::connect                      ~50-200 μs            本地连接
TcpStream::read (有数据)                ~50-100 ns            从缓冲区读
spawn_blocking                          ~5-10 μs              阻塞线程池调度

10K 并发 TCP 连接:
  Tokio (多线程, 8 workers)             ~120 MB 内存          ~15K req/s
  Tokio + io_uring (Linux 5.1+)         ~100 MB 内存          ~25K req/s
  Python asyncio + uvloop               ~200 MB 内存          ~10K req/s
  Go net/http                           ~150 MB 内存          ~18K req/s
```

## 生产案例

### Discord 的 Tokio 迁移

Discord 将核心消息路由从 Go 迁移到 Rust + Tokio：

```
关键收益:
  - 消除了 Go GC 导致的延迟尖峰（200ms → 5ms p99）
  - 内存使用减少 60%
  - CPU 使用率降低 40%
  - 编译期内存安全保证

技术细节:
  - Tokio 多线程运行时，worker 数 = CPU 核心数
  - DashMap 替代 Go 的 sync.Map（分片锁，减少竞争）
  - crossbeam 的无锁队列处理消息路由
  - tracing + tokio-console 用于生产环境监控
```

### Cloudflare 的 Pingora

Cloudflare 用 Rust + Tokio 编写 HTTP 代理，替代 Nginx：

```
Pingora vs Nginx:
  - 内存安全: Rust 编译期保证，Nginx (C) 有大量 CVE
  - 并发模型: Tokio async vs Nginx 事件驱动 + 多进程
  - 可维护性: Rust 类型系统 + Cargo 生态 vs C 手动管理
  - 性能: 相当或更优
```

### Linkerd (Service Mesh)

Linkerd 的数据平面代理 (linkerd2-proxy) 使用 Rust + Tokio：
- 每个 Pod 一个轻量级代理进程
- 处理服务间通信的 TLS、负载均衡、重试
- 内存占用 < 10 MB，启动时间 < 100ms

## 常见陷阱详解

### 陷阱 1：在 async 中使用阻塞操作

```rust
// BUG: 阻塞整个 worker 线程
async fn bad_handler() {
    std::thread::sleep(std::time::Duration::from_secs(1));  // 阻塞!
    // 其他 task 无法在这个 worker 上执行
}

// 修复 1: 使用异步版本
async fn good_handler() {
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;
}

// 修复 2: 必须阻塞时，用 spawn_blocking
async fn ok_handler() {
    tokio::spawn_blocking(|| {
        std::thread::sleep(std::time::Duration::from_secs(1));
    }).await.unwrap();
}
```

### 陷阱 2：持有锁时跨 .await

```rust
use std::sync::{Arc, Mutex};

// BUG: 持有 std::sync::Mutex 时 await
async fn bad_example() {
    let data = Arc::new(Mutex::new(0));
    let mut val = data.lock().unwrap();
    *val += 1;
    // 持有锁时 await → 可能导致死锁
    // 因为另一个 task 也可能在不同的线程尝试获取同一个锁
    tokio::time::sleep(std::time::Duration::from_millis(10)).await;
}

// 修复 1: 缩小锁的作用域
async fn good_example_1() {
    let data = Arc::new(Mutex::new(0));
    {
        let mut val = data.lock().unwrap();
        *val += 1;
    } // 释放锁
    tokio::time::sleep(std::time::Duration::from_millis(10)).await;
}

// 修复 2: 使用 tokio::sync::Mutex
async fn good_example_2() {
    let data = Arc::new(tokio::sync::Mutex::new(0));
    let mut val = data.lock().await;  // 异步锁，支持跨 await
    *val += 1;
    tokio::time::sleep(std::time::Duration::from_millis(10)).await;
}
```

### 陷阱 3：Send 约束违反

```rust
use std::rc::Rc;

// BUG: Future 不是 Send
async fn bad_spawn() {
    let data = Rc::new(42);  // Rc 不是 Send
    tokio::spawn(async move {
        println!("{}", data);  // 编译错误: `Rc<i32>` cannot be sent between threads
    }).await;
}

// 修复 1: 使用 Arc
async fn good_spawn() {
    let data = std::sync::Arc::new(42);  // Arc 是 Send
    tokio::spawn(async move {
        println!("{}", data);
    }).await;
}

// 修复 2: 使用 spawn_local（需要单线程运行时）
#[tokio::main(flavor = "current_thread")]
async fn good_spawn_local() {
    let data = Rc::new(42);
    tokio::task::spawn_local(async move {
        println!("{}", data);
    }).await;
}
```

### 陷阱 4：任务泄漏 — spawn 后不等待 JoinHandle

```rust
// BUG: 任务泄漏
async fn leaky() {
    for i in 0..1000 {
        tokio::spawn(async move {
            tokio::time::sleep(std::time::Duration::from_secs(3600)).await;
            println!("{}", i);
        });
        // 不保存 JoinHandle，任务被"遗忘"
    }
}  // 函数返回，但 1000 个 task 继续运行

// 修复: 收集所有 JoinHandle 并等待
async fn controlled() {
    let mut handles = vec![];
    for i in 0..1000 {
        handles.push(tokio::spawn(async move {
            tokio::time::sleep(std::time::Duration::from_secs(1)).await;
            println!("{}", i);
        }));
    }
    for h in handles {
        h.await.unwrap();
    }
}
```

### 陷阱 5：channel 背压处理不当

```rust
use tokio::sync::mpsc;

// BUG: 无界 channel 可能导致内存爆炸
async fn unbounded_growth() {
    let (tx, mut rx) = mpsc::unbounded_channel();
    // 生产速度 >> 消费速度 → channel 无限增长 → OOM
    tokio::spawn(async move {
        loop {
            tx.send(vec![0u8; 1024]).unwrap();  // 不会阻塞
        }
    });
}

// 修复 1: 使用有界 channel + 处理 Full 错误
async fn bounded_channel() {
    let (tx, mut rx) = mpsc::channel::<Vec<u8>>(100);  // 有界
    tokio::spawn(async move {
        loop {
            if tx.send(vec![0u8; 1024]).await.is_err() {
                break;  // 接收端关闭
            }
        }
    });
}

// 修复 2: 使用 Semaphore 限制生产速度
async fn rate_limited() {
    let (tx, mut rx) = mpsc::channel(100);
    let semaphore = std::sync::Arc::new(tokio::sync::Semaphore::new(10));

    for _ in 0..20 {
        let tx = tx.clone();
        let sem = semaphore.clone();
        tokio::spawn(async move {
            let _permit = sem.acquire().await.unwrap();
            tx.send(vec![0u8; 1024]).await.unwrap();
        });
    }
}
```

## 异步运行时生态对比

```
运行时        线程模型            特色                     适用场景
──────────────────────────────────────────────────────────────────────
Tokio        多线程 work-stealing 功能全面, 生态最大      Web 服务器、网络服务
async-std    多线程               类 std 的 API          学习、小型项目
smol         多线程               极简, 可组合           库开发、嵌入
glommio      单线程 + io_uring    Linux IO 性能极致      高性能存储
monoio       单线程 + io_uring    Thread-per-core        极致性能场景
```

## 总结

| 工具 | 用途 | 说明 |
|------|------|------|
| async fn | 定义异步函数 | 编译为状态机，零成本抽象 |
| .await | 等待 Future | 非阻塞等待，交出控制权 |
| tokio::spawn | 创建异步任务 | 并发执行，要求 Send + 'static |
| select! | 多路选择 | 第一个完成的分支，取消其他 |
| join! / try_join! | 并发等待 | 等待所有完成 |
| oneshot/mpsc/broadcast | 异步 channel | 各有适用场景 |
| tokio::fs | 异步文件 IO | 实际在阻塞线程池执行 |
| spawn_blocking | 阻塞操作 | 在专用线程池运行 |
| tokio-console | 调试监控 | 实时查看 task 状态 |
| tracing | 结构化日志 | 生产级日志和追踪 |
