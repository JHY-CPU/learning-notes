# Atomic原子操作

## 一、概念说明

原子类型提供无锁的线程安全操作。相比 Mutex，原子操作性能更好，但只适用于简单类型。

```rust
use std::sync::atomic::{AtomicI32, Ordering};

let counter = AtomicI32::new(0);
counter.fetch_add(1, Ordering::SeqCst);
println!("{}", counter.load(Ordering::SeqCst));
```

## 二、具体用法

### 2.1 常用原子类型

```rust
use std::sync::atomic::*;

let atomic_bool = AtomicBool::new(true);
let atomic_i32 = AtomicI32::new(0);
let atomic_u64 = AtomicU64::new(0);
let atomic_size = AtomicUsize::new(0);

// 基本操作
atomic_bool.store(false, Ordering::SeqCst);
let val = atomic_i32.load(Ordering::SeqCst);
atomic_u64.fetch_add(1, Ordering::SeqCst);
```

### 2.2 内存序详解

```rust
use std::sync::atomic::Ordering;

// Ordering 选项：
// Relaxed: 最宽松，只保证原子性
// Release: 写操作，之前的操作不会被重排到后面
// Acquire: 读操作，之后的操作不会被重排到前面
// AcqRel: 同时是 Acquire 和 Release
// SeqCst: 最严格，完全顺序一致性

// 典型用法
let flag = AtomicBool::new(false);
let data = AtomicI32::new(0);

// 写入端
data.store(42, Ordering::Relaxed);
flag.store(true, Ordering::Release);

// 读取端
while !flag.load(Ordering::Acquire) {}
let val = data.load(Ordering::Relaxed);
```

### 2.3 CAS 操作

```rust
use std::sync::atomic::{AtomicI32, Ordering};

let counter = AtomicI32::new(0);

// compare_and_swap（已弃用，推荐 compare_exchange）
let result = counter.compare_exchange(
    0,           // 期望值
    1,           // 新值
    Ordering::SeqCst,
    Ordering::SeqCst,
);

match result {
    Ok(old) => println!("成功，旧值: {}", old),
    Err(current) => println!("失败，当前值: {}", current),
}
```

## 三、注意事项与常见陷阱

1. **内存序选择**：大多数场景 SeqCst 足够，复杂场景需仔细选择
2. **ABA 问题**：CAS 可能遇到 ABA 问题
3. **性能权衡**：简单计数器用原子，复杂结构用锁
4. **不可用于引用**：原子类型只支持基本类型
5. **调试困难**：内存序错误难以复现和调试
