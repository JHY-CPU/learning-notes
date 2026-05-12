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

### 2.4 锁竞争度量

```rust
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Instant;

fn measure_contention() {
    let n_threads = 8;
    let iterations = 1_000_000;

    // 高竞争：所有线程竞争同一锁
    let counter = Arc::new(Mutex::new(0u64));
    let start = Instant::now();
    let handles: Vec<_> = (0..n_threads).map(|_| {
        let c = Arc::clone(&counter);
        thread::spawn(move || {
            for _ in 0..iterations / n_threads {
                *c.lock().unwrap() += 1;
            }
        })
    }).collect();
    for h in handles { h.join().unwrap(); }
    println!("高竞争耗时: {:?}", start.elapsed());

    // 低竞争：每个线程独立计数，最后汇总
    let counters: Vec<_> = (0..n_threads).map(|_| Arc::new(Mutex::new(0u64))).collect();
    let start = Instant::now();
    let handles: Vec<_> = counters.iter().map(|c| {
        let c = Arc::clone(c);
        thread::spawn(move || {
            for _ in 0..iterations / n_threads {
                *c.lock().unwrap() += 1;
            }
        })
    }).collect();
    for h in handles { h.join().unwrap(); }
    let total: u64 = counters.iter().map(|c| *c.lock().unwrap()).sum();
    println!("低竞争耗时: {:?}, 总和: {}", start.elapsed(), total);
}
```

### 2.5 细粒度锁设计

```rust
use std::sync::Mutex;
use std::collections::HashMap;

struct ShardedCounter {
    shards: Vec<Mutex<HashMap<String, u64>>>,
}

impl ShardedCounter {
    fn new(n_shards: usize) -> Self {
        ShardedCounter {
            shards: (0..n_shards).map(|_| Mutex::new(HashMap::new())).collect(),
        }
    }

    fn get_shard(&self, key: &str) -> &Mutex<HashMap<String, u64>> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut hasher = DefaultHasher::new();
        key.hash(&mut hasher);
        let idx = (hasher.finish() as usize) % self.shards.len();
        &self.shards[idx]
    }

    fn increment(&self, key: &str) {
        let shard = self.get_shard(key);
        *shard.lock().unwrap().entry(key.to_string()).or_insert(0) += 1;
    }

    fn get(&self, key: &str) -> u64 {
        let shard = self.get_shard(key);
        shard.lock().unwrap().get(key).copied().unwrap_or(0)
    }
}
```

### 2.6 读写锁与写者饥饿

```rust
use std::sync::RwLock;
use std::thread;
use std::time::Duration;

fn rwlock_fairness() {
    let data = Arc::new(RwLock::new(0u64));

    // 注意：std::sync::RwLock 不保证公平性
    // 写者可能饥饿（长时间无法获取写锁）
    // 对于关键场景，考虑使用 parking_lot::RwLock

    let reader_handles: Vec<_> = (0..4).map(|_| {
        let data = Arc::clone(&data);
        thread::spawn(move || {
            for _ in 0..1000 {
                let _val = *data.read().unwrap();
                thread::sleep(Duration::from_micros(1));
            }
        })
    }).collect();

    let writer_handle = {
        let data = Arc::clone(&data);
        thread::spawn(move || {
            for _ in 0..100 {
                *data.write().unwrap() += 1;
                thread::sleep(Duration::from_micros(10));
            }
        })
    };

    for h in reader_handles { h.join().unwrap(); }
    writer_handle.join().unwrap();
}
```

## 四、性能基准对比

| 操作 | Mutex | RwLock | Atomic |
|------|-------|--------|--------|
| 无竞争 | ~25ns | ~30ns | ~5ns |
| 高竞争（8线程） | ~500ns | ~800ns | ~50ns |
| 读多写少 | 一般 | 优秀 | 优秀 |
| 写多 | 一般 | 差 | 优秀 |

## 五、注意事项与常见陷阱

1. **死锁避免**：不要嵌套获取同一锁，使用 try_lock 或统一锁顺序
2. **锁粒度**：锁的持有时间应尽可能短，避免在锁内做耗时操作
3. **panic 安全**：持有锁时 panic 会污染锁，需要处理 poisoned 状态
4. **替代方案**：简单计数器用 AtomicUsize 更高效，复杂状态用分片锁
5. **公平性**：std Mutex 不保证公平性，考虑使用 `parking_lot` crate
6. **Condvar**：条件变量用于线程间等待通知，避免忙等待
7. **超时机制**：使用 `try_lock_for` 或类似 API 避免无限等待
