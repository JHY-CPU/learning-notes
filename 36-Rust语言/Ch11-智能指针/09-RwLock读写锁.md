# RwLock读写锁

## 一、概念说明

`RwLock<T>` 提供了读写锁，允许多个读取者同时访问，但写入时需要独占访问。适用于读多写少的场景。

```rust
use std::sync::{Arc, RwLock};
use std::thread;

let data = Arc::new(RwLock::new(vec![1, 2, 3]));

// 多个线程可以同时读取
let reader = Arc::clone(&data);
let handle = thread::spawn(move || {
    let val = reader.read().unwrap();
    println!("读取: {:?}", *val);
});

// 写入需要独占
let writer = Arc::clone(&data);
let handle2 = thread::spawn(move || {
    let mut val = writer.write().unwrap();
    val.push(4);
});
```

## 二、具体用法

### 2.1 读写分离

```rust
use std::sync::{Arc, RwLock};
use std::thread;

fn reader_writer_example() {
    let data = Arc::new(RwLock::new(vec![1, 2, 3]));
    let mut handles = vec![];

    // 3个读取者
    for i in 0..3 {
        let data = Arc::clone(&data);
        handles.push(thread::spawn(move || {
            let reader = data.read().unwrap();
            println!("读取者{}: {:?}", i, *reader);
        }));
    }

    // 1个写入者
    let data = Arc::clone(&data);
    handles.push(thread::spawn(move || {
        let mut writer = data.write().unwrap();
        writer.push(4);
        println!("写入完成");
    }));

    for h in handles {
        h.join().unwrap();
    }
}
```

### 2.2 缓存示例

```rust
use std::sync::{Arc, RwLock};
use std::collections::HashMap;

struct Cache {
    data: RwLock<HashMap<String, String>>,
}

impl Cache {
    fn new() -> Self {
        Cache { data: RwLock::new(HashMap::new()) }
    }

    fn get(&self, key: &str) -> Option<String> {
        self.data.read().unwrap().get(key).cloned()
    }

    fn set(&self, key: &str, value: &str) {
        self.data.write().unwrap()
            .insert(key.to_string(), value.to_string());
    }
}
```

## 三、注意事项与常见陷阱

1. **死锁风险**：不要在持有读锁时获取写锁
2. **优先级**：写入者通常有优先级
3. **性能**：读多写少时 RwLock 优于 Mutex
4. **Starvation**：连续读取可能饿死写入者
5. **替代方案**：考虑 `ArcSwap` 或 `parking_lot` 的 RwLock
