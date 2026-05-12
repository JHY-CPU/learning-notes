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

### 2.4 锁的 RAII 模式

```rust
use std::sync::Mutex;

fn raii_lock_example() {
    let data = Mutex::new(vec![1, 2, 3]);

    // lock() 返回 MutexGuard，实现了 Deref 和 Drop
    {
        let mut guard = data.lock().unwrap();
        guard.push(4);
        // guard 在作用域结束时自动释放锁
    }

    // 手动 drop 也可以释放锁
    {
        let mut guard = data.lock().unwrap();
        guard.push(5);
        drop(guard); // 立即释放锁
        // 此处可以安全获取另一个锁
    }
}
```

### 2.5 读写锁 RwLock

```rust
use std::sync::RwLock;

fn rwlock_example() {
    let data = RwLock::new(vec![1, 2, 3]);

    // 多个读者可以同时访问
    {
        let reader1 = data.read().unwrap();
        let reader2 = data.read().unwrap();
        println!("读者1: {:?}", *reader1);
        println!("读者2: {:?}", *reader2);
    }

    // 写者独占访问
    {
        let mut writer = data.write().unwrap();
        writer.push(4);
    }
}
```

### 2.6 原子类型替代简单计数器

```rust
use std::sync::atomic::{AtomicUsize, Ordering};

fn atomic_counter_example() {
    let counter = AtomicUsize::new(0);

    // 原子操作，无需锁
    counter.fetch_add(1, Ordering::SeqCst);
    counter.fetch_add(1, Ordering::Relaxed);

    let value = counter.load(Ordering::SeqCst);
    println!("计数: {}", value); // 2

    // 适用场景：简单计数、标志位
    // 比 Mutex 更高效
}
```

### 2.7 条件变量通知

```rust
use std::sync::{Arc, Mutex, Condvar};

fn condvar_example() {
    let pair = Arc::new((Mutex::new(false), Condvar::new()));
    let pair2 = Arc::clone(&pair);

    let handle = std::thread::spawn(move || {
        let (lock, cvar) = &*pair2;
        let mut started = lock.lock().unwrap();
        *started = true;
        cvar.notify_one(); // 通知等待的线程
    });

    let (lock, cvar) = &*pair;
    let mut started = lock.lock().unwrap();
    while !*started {
        started = cvar.wait(started).unwrap();
    }

    handle.join().unwrap();
    println!("工作线程已启动");
}
```

### 2.8 读写比例优化

```rust
use std::sync::RwLock;
use std::collections::HashMap;

struct ReadHeavyCache {
    data: RwLock<HashMap<String, Vec<u8>>>,
}

impl ReadHeavyCache {
    fn new() -> Self {
        ReadHeavyCache { data: RwLock::new(HashMap::new()) }
    }

    // 读操作：允许多个线程同时读
    fn get(&self, key: &str) -> Option<Vec<u8>> {
        self.data.read().unwrap().get(key).cloned()
    }

    // 写操作：独占访问，但读多写少时性能很好
    fn insert(&self, key: String, value: Vec<u8>) {
        self.data.write().unwrap().insert(key, value);
    }

    // 读取并修改（需要短暂的写锁）
    fn get_or_insert(&self, key: &str, default: Vec<u8>) -> Vec<u8> {
        // 先尝试读
        if let Some(value) = self.data.read().unwrap().get(key) {
            return value.clone();
        }
        // 读不到再写
        let mut data = self.data.write().unwrap();
        data.entry(key.to_string())
            .or_insert(default)
            .clone()
    }
}
```

## 四、同步原语选择指南

| 场景 | 推荐类型 | 原因 |
|------|---------|------|
| 简单计数 | AtomicUsize | 无锁，最高效 |
| 保护复杂状态 | Mutex | 简单安全 |
| 读多写少 | RwLock | 允许并发读 |
| 等待条件 | Condvar + Mutex | 避免忙等待 |
| 单次初始化 | OnceLock / LazyLock | 线程安全的延迟初始化 |

## 五、注意事项与常见陷阱

1. **死锁**：同一锁嵌套获取会导致死锁，统一获取顺序可避免
2. **锁粒度**：锁的持有时间应尽可能短，避免在锁内执行耗时操作
3. **panic 传播**：持有锁时 panic 会污染锁（poison），需要处理 poisoned 状态
4. **性能**：竞争激烈时考虑分片（ShardedMap）或无锁算法
5. **RwLock 公平性**：std::sync::RwLock 不保证公平性，可能导致写者饥饿
6. **锁的顺序**：多个锁必须按一致顺序获取，否则会导致死锁
7. **测试覆盖**：并发代码需要压力测试和死锁检测
