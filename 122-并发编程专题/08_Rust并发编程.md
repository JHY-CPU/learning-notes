# Rust 并发编程

## 所有权与并发

Rust 的所有权系统在编译期消除数据竞争，这是 Rust 并发安全的核心保障。无需运行时检查，编译器保证不会出现悬垂引用和数据竞争。

### 所有权转移

```rust
use std::thread;

fn main() {
    let message = String::from("Hello from thread");

    // 所有权转移到新线程
    let handle = thread::spawn(move || {
        println!("{}", message); // message 在这里有效
    });

    // println!("{}", message); // 编译错误：message 已被移动
    handle.join().unwrap();
}
```

### 借用检查器如何保证线程安全

Rust 的借用检查器（Borrow Checker）在编译期执行以下规则：

```
规则                          线程安全含义
──────────────────────────────────────────────────────────────────
同一时刻只有一个可变引用       不存在两个线程同时写
(&mut T)                      → 防止写-写数据竞争
或多个不可变引用 (&T)

不可变引用存在时不能有可变引用  不存在一个线程读的同时另一个线程写
→ 防止读-写数据竞争

所有权转移 (move)              数据从一个线程转移到另一个线程
                              → 转移后原线程无法访问
```

```
编译器检查流程:

fn main() {
    let data = vec![1, 2, 3];            // data 是所有者
    thread::spawn(|| {                    // 闭包捕获 data
        println!("{:?}", data);           // 编译器: data 需要 'static 生命周期
    });                                   // 但 data 是栈上的局部变量
}                                         // → 编译错误!

// 修复: 使用 move 转移所有权
fn main() {
    let data = vec![1, 2, 3];
    thread::spawn(move || {               // data 的所有权转移到闭包
        println!("{:?}", data);           // OK
    });
}
```

### 生命周期与 'static 约束

`thread::spawn` 的签名要求闭包满足 `'static`：

```rust
pub fn spawn<F, T>(f: F) -> JoinHandle<T>
where
    F: FnOnce() -> T + Send + 'static,  // 'static: 不引用栈上的数据
    T: Send + 'static,
```

`'static` 的含义：闭包捕获的值要么是拥有的（owned），要么是 `'static` 引用（如 `&'static str`）。这是为了保证新线程不会引用已经释放的栈变量。

## 编译器如何防止数据竞争

Rust 的数据竞争定义（同时满足以下条件）：
1. 两个或更多线程同时访问同一内存位置
2. 至少一个是写操作
3. 没有同步机制

Rust 通过类型系统在编译期阻止数据竞争：

```rust
use std::thread;

fn main() {
    let mut data = vec![1, 2, 3];

    // 错误: 闭包需要捕获 &mut data，但 main 线程也可能使用 data
    thread::spawn(|| {
        data.push(4);  // 需要 &mut data
    });

    // 如果这里也使用 data，就构成数据竞争
    // data.push(5);  // 编译错误!
}
```

```rust
use std::sync::{Arc, Mutex};
use std::thread;

fn main() {
    // 正确: Arc<Mutex<T>> 提供线程安全的共享可变访问
    let data = Arc::new(Mutex::new(vec![1, 2, 3]));

    let data_clone = Arc::clone(&data);
    thread::spawn(move || {
        data_clone.lock().unwrap().push(4);
    });

    data.lock().unwrap().push(5);
    // Mutex 保证互斥，Arc 保证线程安全的引用计数
}
```

## Arc / Mutex

`Arc`（Atomic Reference Counting）提供线程安全的引用计数，`Mutex` 提供互斥访问。

```rust
use std::sync::{Arc, Mutex};
use std::thread;

fn main() {
    // Arc<Mutex<T>> 是 Rust 中最常见的线程安全共享状态模式
    let counter = Arc::new(Mutex::new(0));
    let mut handles = vec![];

    for i in 0..10 {
        let counter = Arc::clone(&counter);
        let handle = thread::spawn(move || {
            let mut num = counter.lock().unwrap();
            *num += 1;
            println!("线程 {} 计数: {}", i, *num);
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }

    println!("最终计数: {}", *counter.lock().unwrap()); // 10
}
```

### Arc 的底层实现

```
Arc<T> 的内存布局:

栈上                 堆上
┌──────────┐    ┌──────────────────────┐
│ Arc 指针  │───►│ strong_count: usize  │  ← 原子引用计数
│          │    │ weak_count: usize    │  ← 弱引用计数
│          │    │ data: T              │  ← 实际数据
└──────────┘    └──────────────────────┘

Arc::clone():
  不复制数据，只复制指针并原子地增加 strong_count
  时间复杂度: O(1)

Drop(Arc):
  原子地减少 strong_count
  若 strong_count == 0，释放数据
```

```
Rc<T> vs Arc<T>:

类型      引用计数操作      线程安全    性能
──────────────────────────────────────────────
Rc<T>    普通整数操作       否         ~0 ns
Arc<T>   原子操作           是         ~10 ns
```

### Mutex 的底层实现

```
Mutex<T> 的内部结构:

pub struct Mutex<T> {
    inner: sys::Mutex,     // 平台特定的互斥锁实现
    poison: poison::Flag,  // 检测 panic 导致的锁状态不一致
    data: UnsafeCell<T>,   // 内部可变性容器
}

Linux 实现:
  使用 pthread_mutex_t (futex 在无竞争时)
  无竞争路径: 单条原子 CAS 指令 (~17 ns)
  有竞争路径: futex 系统调用 → 内核休眠 (~200 ns+)

Guard (MutexGuard<T>):
  - 实现了 Deref 和 DerefMut
  - Drop 时自动释放锁
  - 实现了 !Send（不能跨线程转移）
  - 实现了 Sync（如果 T 是 Sync）
```

### 读写锁 RwLock

```rust
use std::sync::{Arc, RwLock};
use std::thread;

fn main() {
    let data = Arc::new(RwLock::new(vec![1, 2, 3]));
    let mut handles = vec![];

    // 多个读线程可以并发访问
    for i in 0..3 {
        let data = Arc::clone(&data);
        handles.push(thread::spawn(move || {
            let reader = data.read().unwrap();
            println!("读线程 {}: {:?}", i, *reader);
        }));
    }

    // 写线程独占访问
    let data_clone = Arc::clone(&data);
    handles.push(thread::spawn(move || {
        let mut writer = data_clone.write().unwrap();
        writer.push(4);
        println!("写入: {:?}", *writer);
    }));

    for h in handles {
        h.join().unwrap();
    }
}
```

### Arc<Mutex<T>> vs Arc<RwLock<T>> 性能对比

```
场景                        Arc<Mutex<T>>     Arc<RwLock<T>>
────────────────────────────────────────────────────────────────
读多写少 (99% 读)            所有操作串行       读操作并行
读写各半                     相当              相当
写多读少                     更好              额外开销
简单计数器                    更好              过度设计

无竞争基准 (x86-64):
  Mutex::lock()              ~17 ns
  RwLock::read()             ~20 ns
  RwLock::write()            ~22 ns

高竞争基准 (8 线程):
  Mutex::lock()              ~200 ns
  RwLock::read() (全读)       ~30 ns   ← 读并行优势明显
  RwLock::write()             ~250 ns  ← 写竞争更激烈
```

## Channels（mpsc）

`mpsc`（Multiple Producer, Single Consumer）是 Rust 标准库提供的多生产者单消费者 channel。

### 基本用法

```rust
use std::sync::mpsc;
use std::thread;

fn main() {
    let (tx, rx) = mpsc::channel();

    // 在新线程中发送数据
    thread::spawn(move || {
        let msg = String::from("你好");
        tx.send(msg).unwrap();
        // println!("{}", msg); // 编译错误：msg 已被移动
    });

    // 主线程接收数据
    let received = rx.recv().unwrap();
    println!("收到: {}", received);
}
```

### 多生产者

```rust
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

fn main() {
    let (tx, rx) = mpsc::channel();

    // 克隆发送端，实现多生产者
    for i in 0..5 {
        let tx_clone = tx.clone();
        thread::spawn(move || {
            thread::sleep(Duration::from_millis(i * 100));
            tx_clone.send(format!("消息来自线程 {}", i)).unwrap();
        });
    }

    // 必须 drop 原始 tx，否则 rx 不会收到结束信号
    drop(tx);

    // 接收所有消息
    for received in rx {
        println!("收到: {}", received);
    }
}
```

### 同步 channel

```rust
use std::sync::mpsc;
use std::thread;

fn main() {
    // sync_channel(0): 发送会阻塞直到接收方准备好
    // sync_channel(n): 缓冲区大小为 n
    let (tx, rx) = mpsc::sync_channel(2);

    thread::spawn(move || {
        for i in 0..5 {
            println!("发送: {}", i);
            tx.send(i).unwrap(); // 缓冲区满时会阻塞
        }
    });

    thread::sleep(std::time::Duration::from_millis(500));

    for received in rx {
        println!("接收: {}", received);
    }
}
```

### 传输结构体

```rust
use std::sync::mpsc;
use std::thread;

struct Task {
    id: u32,
    data: String,
}

fn main() {
    let (tx, rx) = mpsc::channel();

    thread::spawn(move || {
        for i in 0..3 {
            tx.send(Task {
                id: i,
                data: format!("任务数据-{}", i),
            }).unwrap();
        }
    });

    for task in rx {
        println!("处理任务 {}: {}", task.id, task.data);
    }
}
```

### Channel 的内部实现

```
mpsc::channel 内部结构 (简化):

channel<T> {
    // 共享状态（所有 tx 和 rx 共享）
    shared: AtomicPtr<Shared<T>>,

    // 发送端私有
    tail: UnsafeCell<*mut Packet<T>>,   // 生产者链表尾

    // 接收端私有
    head: UnsafeCell<*mut Packet<T>>,   // 消费者链表头
}

共享状态 {
    queue: spin::Mutex<...>,     // 有界队列（如果是有缓冲）
    to_wake: AtomicUsize,         // 唤醒标记
    channel_lock: Mutex,          // 接收端休眠用
}

发送流程 (无竞争):
  1. CAS 更新 tail 指针
  2. 写入数据
  3. 如果有等待的接收者，唤醒它

接收流程:
  1. 尝试从 head 取数据（无锁快速路径）
  2. 如果没有数据，进入休眠等待
```

## Send 和 Sync Trait

`Send` 表示类型可以安全地在线程间转移所有权，`Sync` 表示类型可以安全地被多个线程引用。

```rust
// Send: 可以安全地转移所有权到另一个线程
// 大多数类型自动实现 Send
// Rc<T> 不是 Send（非原子引用计数）

// Sync: 可以安全地被多个线程通过 &T 共享
// 如果 T 是 Sync，则 &T 是 Send

// 以下类型自动满足 Send + Sync:
// - 所有基本类型（i32, bool, f64, etc.）
// - 仅包含 Send 类型的元组和结构体
// - &T（当 T 是 Sync 时）

// 以下类型不是 Send:
// - Rc<T>
// - Cell<T> 和 RefCell<T>（不保证线程安全）
// - 指针类型

// 手动标记 Send + Sync（通常通过 unsafe）
// 通常不需要手动实现，编译器会自动推导
use std::sync::Arc;

fn assert_send<T: Send>() {}
fn assert_sync<T: Sync>() {}

fn main() {
    assert_send::<Arc<i32>>();   // Arc<T> 是 Send
    assert_sync::<Arc<i32>>();   // Arc<T> 是 Sync
    println!("Send + Sync 检查通过");
}
```

### Send/Sync 的自动推导规则

```
类型                    Send?    Sync?    原因
──────────────────────────────────────────────────────────────
i32, String, Vec<T>     ✓        ✓       值语义，无共享状态
&T (T: Sync)            ✓        ✓       不可变共享引用
&mut T (T: Send)        ✓        ✗       可变引用不能共享
Rc<T>                   ✗        ✗       非原子引用计数
Arc<T>                  ✓        ✓       原子引用计数
Cell<T>                 ✓        ✗       非线程安全的内部可变性
RefCell<T>              ✓        ✗       非线程安全的动态借用检查
Mutex<T>                ✓        ✓       互斥访问保护
UnsafeCell<T>           ✓ (if T: Send)  ✗  内部可变性原语
```

Send 和 Sync 是 auto trait（自动推导），大多数情况下不需要手动实现。手动实现需要 `unsafe`：

```rust
// 自定义类型示例
struct MyType {
    raw_ptr: *mut u8,
}

// 危险: 如果内部指针跨线程不安全，不应该实现 Send
// unsafe impl Send for MyType {}
// unsafe impl Sync for MyType {}

// 安全的做法: 包装为线程安全类型
use std::sync::Arc;
struct SafeWrapper(Arc<Inner>);
// 自动推导 Send + Sync（如果 Inner 是 Send + Sync）
```

## Rayon 并行计算库

Rayon 提供了数据并行的高层抽象，类似串行迭代器但自动并行化。

### 基本用法

```rust
use rayon::prelude::*;

fn main() {
    // 并行迭代器
    let sum: i64 = (1..=1_000_000)
        .into_par_iter()
        .map(|i| i * i)
        .sum();
    println!("平方和: {}", sum);

    // 并行排序
    let mut nums = vec![5, 3, 8, 1, 9, 2, 7, 4, 6];
    nums.par_sort();
    println!("排序: {:?}", nums);

    // 并行 filter + map
    let results: Vec<i32> = (0..100)
        .into_par_iter()
        .filter(|&x| x % 2 == 0)
        .map(|x| x * x)
        .collect();
    println!("偶数平方: {:?}", results.len());
}
```

### par_iter 与 par_iter_mut

```rust
use rayon::prelude::*;

fn process_image(pixels: &mut [u8]) {
    // 并行修改每个像素
    pixels.par_iter_mut().for_each(|pixel| {
        *pixel = pixel.saturating_add(10);
    });
}

fn main() {
    // par_iter: 并行只读遍历
    let data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    let doubled: Vec<i32> = data.par_iter()
        .map(|&x| x * 2)
        .collect();
    println!("翻倍: {:?}", doubled);

    // par_chunks: 并行处理块
    let big_data: Vec<i64> = (0..1_000_000).collect();
    let chunk_sums: Vec<i64> = big_data.par_chunks(1000)
        .map(|chunk| chunk.iter().sum())
        .collect();
    println!("块总和数量: {}", chunk_sums.len());
}
```

### Rayon 的底层工作原理

```
Rayon 使用工作窃取调度器:

全局任务队列
    │
    ├── 线程 0: [子任务, 子任务, ...]
    │           ├── 执行本地任务
    │           └── 空闲时窃取其他线程的任务
    │
    ├── 线程 1: [子任务, 子任务, ...]
    │           └── ...
    │
    └── 线程 N: [...]

par_iter() 的工作方式:
  1. 将数据分成若干块（通常 = 线程数 × 4）
  2. 每个块作为一个任务
  3. 任务被分配到线程的本地队列
  4. 工作窃取保证负载均衡

分块策略:
  - 大块: 更少的调度开销，但可能负载不均
  - 小块: 更好的负载均衡，但更多调度开销
  - Rayon 自动选择: 每个线程至少 4 个任务
```

### Rayon 的 join 操作

```rust
use rayon::join;

fn fibonacci(n: u32) -> u32 {
    if n < 2 {
        return n;
    }
    // 并行计算斐波那契数列
    let (a, b) = join(
        || fibonacci(n - 1),  // 左分支
        || fibonacci(n - 2),  // 右分支（可能并行执行）
    );
    a + b
}

fn main() {
    // 但注意：直接用 Rayon 并行 fibonacci 效率很差
    // 因为任务粒度太细，调度开销远大于计算开销
    // 正确做法：设定阈值，小任务串行执行

    let result = fibonacci(30);
    println!("fib(30) = {}", result);
}

// 更实用的并行分治
fn parallel_sum(data: &[i64]) -> i64 {
    if data.len() < 1000 {
        // 小数组直接串行计算
        return data.iter().sum();
    }
    let mid = data.len() / 2;
    let (left, right) = data.split_at(mid);
    let (a, b) = rayon::join(
        || parallel_sum(left),
        || parallel_sum(right),
    );
    a + b
}
```

## atomic 原子类型

```rust
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;

fn main() {
    let counter = Arc::new(AtomicUsize::new(0));
    let mut handles = vec![];

    for _ in 0..10 {
        let counter = Arc::clone(&counter);
        handles.push(thread::spawn(move || {
            for _ in 0..1000 {
                counter.fetch_add(1, Ordering::SeqCst);
            }
        }));
    }

    for h in handles {
        h.join().unwrap();
    }

    println!("最终计数: {}", counter.load(Ordering::SeqCst)); // 10000
}
```

### 内存顺序（Memory Ordering）详解

Rust 提供了 6 种内存顺序，对应 C++11 的内存模型：

```
顺序                   强度    保证
──────────────────────────────────────────────────────────────────────
Relaxed                最弱    只保证原子性，不保证顺序
Acquire                中等    之后的读写不能重排到此之前
Release                中等    之前的读写不能重排到此之后
AcqRel                 强      Acquire + Release（用于 read-modify-write）
SeqCst                 最强    完全顺序一致性（默认）
Consume                特殊    数据依赖（Rust 目前等同于 Acquire）
```

```
实际使用场景:

场景                          推荐顺序           说明
──────────────────────────────────────────────────────────────────
简单计数器                    Relaxed            只需要最终结果正确
标志位 (flag)                 Acquire/Release    确保数据可见性
双重检查锁                    Acquire/Release    DCL 模式
无锁数据结构                  AcqRel             CAS 操作
全局事件顺序重要时             SeqCst             性能最差但最安全
```

```rust
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::thread;

// Relaxed: 只关心最终值
fn relaxed_example() {
    use std::sync::atomic::AtomicI32;
    let counter = AtomicI32::new(0);
    thread::scope(|s| {
        for _ in 0..10 {
            s.spawn(|| {
                for _ in 0..1000 {
                    counter.fetch_add(1, Ordering::Relaxed);
                }
            });
        }
    });
    println!("Relaxed 计数: {}", counter.load(Ordering::Relaxed));
}

// Acquire/Release: 保证数据可见性
fn acquire_release_example() {
    let data = AtomicU64::new(0);
    let ready = AtomicBool::new(false);

    // 生产者
    thread::scope(|s| {
        s.spawn(|| {
            data.store(42, Ordering::Relaxed);
            ready.store(true, Ordering::Release);  // Release: 之前的写入对 Acquire 可见
        });

        // 消费者
        s.spawn(|| {
            while !ready.load(Ordering::Acquire) {  // Acquire: 之后的读取能看到 Release 之前
                thread::yield_now();
            }
            println!("数据: {}", data.load(Ordering::Relaxed));  // 保证看到 42
        });
    });
}
```

### SeqCst vs Relaxed 性能对比

```
操作                   SeqCst        Relaxed       差异
────────────────────────────────────────────────────────────
load                  ~8 ns         ~5 ns         1.6x
store                 ~10 ns        ~5 ns         2x
fetch_add (无竞争)    ~12 ns        ~8 ns         1.5x
fetch_add (8线程竞争)  ~150 ns       ~100 ns       1.5x

x86-64 上差异较小（因为 x86 是强内存模型）
ARM 上差异显著（弱内存模型，SeqCst 需要额外屏障）
```

## std::thread 高级用法

```rust
use std::thread;
use std::time::Duration;

fn main() {
    // 线程命名
    let handle = thread::Builder::new()
        .name("worker-1".into())
        .stack_size(4 * 1024 * 1024) // 4MB 栈
        .spawn(|| {
            println!("线程名: {:?}", thread::current().name());
        })
        .unwrap();

    handle.join().unwrap();

    // scope: 允许借用栈数据
    let data = vec![1, 2, 3];
    thread::scope(|s| {
        s.spawn(|| {
            println!("借用数据: {:?}", &data);
        });
        s.spawn(|| {
            println!("另一个线程: {:?}", &data);
        });
    }); // scope 结束时等待所有线程完成
}
```

### thread::scope 的底层原理

```rust
// thread::scope (Rust 1.63+) 使得借用栈数据成为可能
fn main() {
    let mut data = vec![1, 2, 3];

    thread::scope(|s| {
        s.spawn(|| {
            // 可以借用 data，不需要 move
            // 编译器保证: scope 结束时所有线程都已结束
            // 所以 data 的引用在此期间始终有效
            println!("读取: {:?}", &data);
        });

        s.spawn(|| {
            // 错误: 不能同时有可变引用和不可变引用
            // data.push(4);
        });
    });

    // scope 结束后，data 仍然有效
    data.push(4);
}
```

`thread::scope` 的关键：编译器通过生命周期推断，保证了作用域内线程的生命周期不超过传入的闭包。这样就可以安全地借用栈上的数据。

## Condvar 条件变量

```rust
use std::sync::{Arc, Mutex, Condvar};
use std::thread;

fn main() {
    let pair = Arc::new((Mutex::new(false), Condvar::new()));
    let pair2 = Arc::clone(&pair);

    thread::spawn(move || {
        let (lock, cvar) = &*pair2;
        let mut started = lock.lock().unwrap();
        *started = true;
        cvar.notify_one(); // 通知等待的线程
    });

    let (lock, cvar) = &*pair;
    let mut started = lock.lock().unwrap();
    while !*started {
        started = cvar.wait(started).unwrap(); // 等待通知
    }
    println!("工作开始!");
}
```

### Condvar 的常见用法：生产者-消费者

```rust
use std::sync::{Arc, Mutex, Condvar};
use std::thread;
use std::collections::VecDeque;

fn main() {
    let queue = Arc::new((Mutex::new(VecDeque::new()), Condvar::new()));
    let mut handles = vec![];

    // 3 个消费者
    for id in 0..3 {
        let queue = Arc::clone(&queue);
        handles.push(thread::spawn(move || {
            let (lock, cvar) = &*queue;
            loop {
                let mut q = lock.lock().unwrap();
                while q.is_empty() {
                    // wait 会释放锁并休眠，被唤醒后重新获取锁
                    q = cvar.wait(q).unwrap();
                }
                let item = q.pop_front().unwrap();
                drop(q); // 及时释放锁

                println!("消费者 {} 处理: {}", id, item);
                if item == -1 {
                    break; // 结束信号
                }
            }
        }));
    }

    // 生产者
    let (lock, cvar) = &*queue;
    for i in 0..10 {
        let mut q = lock.lock().unwrap();
        q.push_back(i);
        cvar.notify_one(); // 唤醒一个等待的消费者
    }

    // 发送结束信号
    for _ in 0..3 {
        let mut q = lock.lock().unwrap();
        q.push_back(-1);
        cvar.notify_one();
    }

    for h in handles {
        h.join().unwrap();
    }
}
```

## 完整工程级示例：并发安全的 LRU 缓存

```rust
use std::collections::HashMap;
use std::hash::Hash;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

struct LruEntry<V> {
    value: V,
    last_access: Instant,
    ttl: Duration,
}

pub struct ConcurrentLruCache<K, V> {
    data: RwLock<HashMap<K, LruEntry<V>>>,
    capacity: usize,
}

impl<K, V> ConcurrentLruCache<K, V>
where
    K: Eq + Hash + Clone,
    V: Clone,
{
    pub fn new(capacity: usize) -> Self {
        Self {
            data: RwLock::new(HashMap::with_capacity(capacity)),
            capacity,
        }
    }

    pub fn get(&self, key: &K) -> Option<V> {
        let mut data = self.data.write().unwrap();
        if let Some(entry) = data.get_mut(key) {
            if entry.last_access.elapsed() > entry.ttl {
                data.remove(key);
                return None;
            }
            entry.last_access = Instant::now();
            Some(entry.value.clone())
        } else {
            None
        }
    }

    pub fn insert(&self, key: K, value: V, ttl: Duration) {
        let mut data = self.data.write().unwrap();

        // 容量满时淘汰最久未访问的条目
        if data.len() >= self.capacity && !data.contains_key(&key) {
            if let Some(oldest_key) = data
                .iter()
                .min_by_key(|(_, entry)| entry.last_access)
                .map(|(k, _)| k.clone())
            {
                data.remove(&oldest_key);
            }
        }

        data.insert(key, LruEntry {
            value,
            last_access: Instant::now(),
            ttl,
        });
    }

    pub fn len(&self) -> usize {
        self.data.read().unwrap().len()
    }

    pub fn evict_expired(&self) -> usize {
        let mut data = self.data.write().unwrap();
        let before = data.len();
        data.retain(|_, entry| entry.last_access.elapsed() <= entry.ttl);
        before - data.len()
    }
}

fn main() {
    let cache = Arc::new(ConcurrentLruCache::new(1000));
    let mut handles = vec![];

    // 8 个线程并发读写
    for i in 0..8 {
        let cache = Arc::clone(&cache);
        handles.push(std::thread::spawn(move || {
            for j in 0..1000 {
                let key = format!("key-{}-{}", i, j);
                cache.insert(key.clone(), j, Duration::from_secs(60));
                let _ = cache.get(&key);
            }
        }));
    }

    for h in handles {
        h.join().unwrap();
    }

    println!("缓存大小: {}", cache.len());
    let evicted = cache.evict_expired();
    println!("淘汰过期条目: {}", evicted);
}
```

## 无锁并发数据结构

### 无锁队列（基于 Crossbeam）

```rust
// crossbeam 提供了生产级的无锁数据结构
use crossbeam::queue::ArrayQueue;
use std::sync::Arc;
use std::thread;

fn main() {
    let queue = Arc::new(ArrayQueue::new(1000));
    let mut handles = vec![];

    // 生产者
    for i in 0..4 {
        let queue = Arc::clone(&queue);
        handles.push(thread::spawn(move || {
            for j in 0..250 {
                while queue.push(i * 250 + j).is_err() {
                    thread::yield_now();
                }
            }
        }));
    }

    // 消费者
    let mut consumers = vec![];
    for _ in 0..2 {
        let queue = Arc::clone(&queue);
        consumers.push(thread::spawn(move || {
            let mut count = 0;
            loop {
                match queue.pop() {
                    Some(_) => count += 1,
                    None => {
                        if count > 0 {
                            return count;
                        }
                        thread::yield_now();
                    }
                }
            }
        }));
    }

    for h in handles {
        h.join().unwrap();
    }
    let total: usize = consumers.into_iter().map(|h| h.join().unwrap()).sum();
    println!("消费总数: {}", total);
}
```

## 调试方法论

### 1. 使用 cargo miri 检测未定义行为

```bash
# Miri: Rust 的 UB 检测器
# 可以检测: 内存泄漏、use-after-free、数据竞争等
rustup component add miri
cargo miri test
cargo miri run

# 检测特定测试
cargo miri test test_concurrent_code

# 环境变量配置
MIRIFLAGS="-Zmiri-disable-isolation" cargo miri run
# -Zmiri-disable-isolation: 允许访问外部资源（如文件系统）
# -Zmiri-track-alloc-id=<id>: 跟踪特定分配
```

### 2. 使用 ThreadSanitizer (TSan) 检测数据竞争

```bash
# 需要 nightly 工具链
rustup toolchain install nightly
RUSTFLAGS="-Z sanitizer=thread" cargo +nightly run --target x86_64-unknown-linux-gnu

# TSan 可以检测到:
# - 未加锁的数据竞争
# - 信号量使用错误
# - 对已释放内存的并发访问

# 示例输出:
# WARNING: ThreadSanitizer: data race (pid=12345)
#   Write of size 4 at 0x7f8b by thread T1:
#     #0 main::{{closure}} src/main.rs:15:9
#   Previous write of size 4 at 0x7f8b by thread T2:
#     #0 main::{{closure}} src/main.rs:20:9
```

### 3. 使用 Loom 进行并发测试

```rust
// loom: 系统化并发测试工具
// 遍历所有可能的线程调度顺序
#[cfg(loom)]
use loom::sync::{Arc, Mutex};
#[cfg(loom)]
use loom::thread;

#[cfg(not(loom))]
use std::sync::{Arc, Mutex};
#[cfg(not(loom))]
use std::thread;

fn increment_counter(counter: &Arc<Mutex<i32>>) {
    let mut lock = counter.lock().unwrap();
    *lock += 1;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_concurrent_increment() {
        // loom 会探索所有可能的交错
        loom::model(|| {
            let counter = Arc::new(Mutex::new(0));
            let c1 = Arc::clone(&counter);
            let c2 = Arc::clone(&counter);

            let t1 = thread::spawn(move || {
                increment_counter(&c1);
            });
            let t2 = thread::spawn(move || {
                increment_counter(&c2);
            });

            t1.join().unwrap();
            t2.join().unwrap();

            assert_eq!(*counter.lock().unwrap(), 2);
        });
    }
}
```

### 4. 常用调试命令

```bash
# 检查死锁
RUST_BACKTRACE=1 cargo run
# 如果死锁，程序会挂起，使用 Ctrl+C 并查看 backtrace

# 性能分析
cargo build --release
perf record --call-graph=dwarf ./target/release/myapp
perf report

# 查看生成的汇编
cargo rustc --release -- --emit asm
# 或使用 godbolt.org 在线查看

# 检查未使用的依赖
cargo machete

# 检查 unsafe 代码
cargo geiger
```

## 生产案例

### Discord 的 Rust 并发实践

Discord 将消息路由器从 Go 迁移到 Rust，获得了显著的性能提升：

```
迁移前 (Go)                          迁移后 (Rust)
──────────────────────────────────────────────────────────────
GC 停顿导致延迟尖峰 (~200ms)          无 GC，延迟稳定 (~5ms)
goroutine 栈占用大量内存               线程池 + async，内存减半
运行时不可预测行为                     编译期保证的确定性

关键技术:
  - Tokio 异步运行时
  - DashMap (分片并发 HashMap) 替代 sync.Map
  - Arc<Mutex<>> / Arc<RwLock<>> 共享状态
  - crossbeam 无锁队列用于消息路由
```

### Figma 的 Rust 并发服务

Figma 使用 Rust 编写多人协作服务器：
- 利用 Rust 的所有权系统保证并发正确性
- 使用 Tokio 处理 WebSocket 连接
- `Arc<RwLock<>>` 管理共享文档状态
- 相比 Node.js 版本，CPU 使用率降低 10 倍

### Cloudflare 的 Rust 代理

Cloudflare 使用 Rust 编写 HTTP 代理（Pingora）：
- 替代 Nginx (C)，获得更好的并发性能
- Tokio 处理百万级并发连接
- 编译期内存安全保证，避免 C 的内存安全漏洞

## 常见陷阱详解

### 陷阱 1：死锁 — 同一线程重复获取 Mutex

```rust
use std::sync::{Arc, Mutex};

// BUG: 死锁 - 同一线程尝试两次获取同一个 Mutex
fn deadlock_example() {
    let data = Arc::new(Mutex::new(0));

    let mut val = data.lock().unwrap();
    // 如果在持有锁的情况下调用一个需要获取同一锁的函数
    // let val2 = data.lock().unwrap(); // 死锁!

    // Rust 的 std::sync::Mutex 不支持递归锁
    // 如果当前线程已经持有锁，再次 lock() 会死锁
    *val += 1;
}

// 修复方案: 避免嵌套锁，或使用 lock API 作用域
fn safe_example() {
    let data = Arc::new(Mutex::new(0));

    {
        let mut val = data.lock().unwrap();
        *val += 1;
    } // 锁在这里释放

    {
        let val = data.lock().unwrap();
        println!("值: {}", *val);
    }
}
```

### 陷阱 2：PoisonError — panic 导致锁中毒

```rust
use std::sync::{Arc, Mutex};
use std::thread;

// BUG: 持有锁的线程 panic，导致锁被"毒化"
fn poison_example() {
    let data = Arc::new(Mutex::new(vec![1, 2, 3]));
    let data_clone = Arc::clone(&data);

    let handle = thread::spawn(move || {
        let mut val = data_clone.lock().unwrap();
        val.push(4);
        panic!("出错了!"); // panic 时锁被标记为 poisoned
    });

    handle.join().unwrap_err();

    // 之后获取锁会返回 Err(PoisonError)
    match data.lock() {
        Ok(mut val) => {
            // 可以通过 into_inner() 获取内部数据
            val.push(5);
        }
        Err(poisoned) => {
            // 恢复: 获取被毒化的 guard
            let mut val = poisoned.into_inner();
            val.push(5);
            println!("从毒化锁恢复: {:?}", *val);
        }
    }
}

// 修复: 在可能 panic 的代码中使用 catch_unwind
fn safe_example() {
    let data = Arc::new(Mutex::new(vec![1, 2, 3]));
    let data_clone = Arc::clone(&data);

    let handle = thread::spawn(move || {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut val = data_clone.lock().unwrap();
            val.push(4);
            panic!("出错了!");
        }));
        // 锁在 panic 后通过 RAII 自动释放
        result.ok()
    });
    handle.join().unwrap();
}
```

### 陷阱 3：忘记 drop 导致锁持有时间过长

```rust
use std::sync::{Arc, Mutex};

// BUG: 锁持有时间过长，阻塞其他线程
fn long_lock() {
    let data = Arc::new(Mutex::new(vec![]));

    let val = data.lock().unwrap();
    // 在持有锁的情况下执行耗时操作
    std::thread::sleep(std::time::Duration::from_secs(1));
    println!("{:?}", *val);
    // 锁在 val 离开作用域时才释放
}

// 修复: 最小化锁的作用域
fn short_lock() {
    let data = Arc::new(Mutex::new(vec![]));

    let snapshot = {
        let val = data.lock().unwrap();
        val.clone() // 复制数据，尽快释放锁
    }; // 锁在这里释放

    // 不持有锁的情况下处理数据
    std::thread::sleep(std::time::Duration::from_secs(1));
    println!("{:?}", snapshot);
}
```

### 陷阱 4：Arc 循环引用导致内存泄漏

```rust
use std::sync::{Arc, Mutex};

// BUG: Arc 循环引用导致内存泄漏
fn arc_cycle() {
    struct Node {
        name: String,
        children: Mutex<Vec<Arc<Node>>>,
        parent: Mutex<Option<Arc<Node>>>,
    }

    let parent = Arc::new(Node {
        name: "parent".to_string(),
        children: Mutex::new(vec![]),
        parent: Mutex::new(None),
    });

    let child = Arc::new(Node {
        name: "child".to_string(),
        children: Mutex::new(vec![]),
        parent: Mutex::new(Some(Arc::clone(&parent))), // child -> parent
    });

    parent.children.lock().unwrap().push(Arc::clone(&child)); // parent -> child

    // parent 引用 child, child 引用 parent
    // 引用计数永远不会降到 0，内存泄漏!
}

// 修复: 使用 Weak 打破循环
use std::sync::Weak;

fn fixed_cycle() {
    struct Node {
        name: String,
        children: Mutex<Vec<Arc<Node>>>,
        parent: Mutex<Option<Weak<Node>>>,  // 使用 Weak
    }

    let parent = Arc::new(Node {
        name: "parent".to_string(),
        children: Mutex::new(vec![]),
        parent: Mutex::new(None),
    });

    let child = Arc::new(Node {
        name: "child".to_string(),
        children: Mutex::new(vec![]),
        parent: Mutex::new(Some(Arc::downgrade(&parent))), // Weak 引用
    });

    parent.children.lock().unwrap().push(Arc::clone(&child));

    // 使用 Weak 引用时需要 upgrade() 检查是否还存在
    if let Some(parent_ref) = child.parent.lock().unwrap().as_ref() {
        if let Some(p) = parent_ref.upgrade() {
            println!("parent: {}", p.name);
        }
    }
    // 离开作用域后正确释放
}
```

## 性能实测数据

```
操作                               耗时 (x86-64)          说明
──────────────────────────────────────────────────────────────────────
thread::spawn                      ~15 μs                 创建 OS 线程
thread::scope                      ~15 μs                 创建 OS 线程
Mutex::lock (无竞争)               ~17 ns                 futex 快速路径
Mutex::lock (有竞争)               ~200 ns - 2 μs         含系统调用
RwLock::read (无竞争)              ~20 ns                 原子操作
RwLock::write (无竞争)             ~22 ns                 原子操作
Arc::clone                         ~8 ns                  原子加
Arc::drop                          ~10 ns                 原子减
AtomicUsize::fetch_add (Relaxed)   ~5 ns                  单条 CPU 指令
AtomicUsize::fetch_add (SeqCst)    ~8 ns                  含内存屏障
mpsc::channel send (无竞争)        ~40 ns                 链表操作
mpsc::channel recv (无竞争)        ~35 ns                 链表操作
Rayon par_sort (1M 元素)           ~12 ms                 8 线程
标准 sort (1M 元素)                ~45 ms                 单线程
```

## 总结

| 工具 | 用途 | 安全保证 |
|------|------|---------|
| 所有权 + 生命周期 | 编译期安全 | 零运行时开销 |
| Send + Sync trait | 并发类型安全 | 编译期推导 |
| Arc + Mutex | 共享可变状态 | 编译期防止数据竞争 |
| RwLock | 读写锁 | 多读单写 |
| mpsc::channel | 消息传递 | 类型安全的通信 |
| Rayon | 数据并行 | 自动线程池管理 |
| Atomic | 无锁操作 | 硬件级原子性 |
| thread::scope | 作用域并发 | 借用栈数据安全 |
| Condvar | 条件等待 | 配合 Mutex 使用 |
