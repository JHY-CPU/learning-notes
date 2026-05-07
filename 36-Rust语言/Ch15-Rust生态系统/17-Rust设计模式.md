# Rust设计模式

## 一、概念说明

Rust 有独特的设计模式，充分利用所有权、trait 和枚举。

```rust
// Builder 模式
struct ServerBuilder {
    host: String,
    port: u16,
}

impl ServerBuilder {
    fn new() -> Self { todo!() }
    fn host(mut self, host: &str) -> Self { todo!() }
    fn port(mut self, port: u16) -> Self { todo!() }
    fn build(self) -> Server { todo!() }
}
```

## 二、具体用法

### 2.1 Newtype 模式

```rust
// 类型安全的包装
struct UserId(u64);
struct Email(String);

impl Email {
    fn new(email: &str) -> Result<Self, String> {
        if email.contains('@') {
            Ok(Email(email.to_string()))
        } else {
            Err("无效邮箱".to_string())
        }
    }
}
```

### 2.2 Typestate 模式

```rust
struct Locked;
struct Unlocked;

struct Door<State> {
    _state: std::marker::PhantomData<State>,
}

impl Door<Locked> {
    fn unlock(self) -> Door<Unlocked> {
        Door { _state: std::marker::PhantomData }
    }
}

impl Door<Unlocked> {
    fn lock(self) -> Door<Locked> {
        Door { _state: std::marker::PhantomData }
    }
}
```

### 2.3 RAII 模式

```rust
struct FileGuard {
    file: std::fs::File,
}

impl Drop for FileGuard {
    fn drop(&mut self) {
        println!("关闭文件");
    }
}

impl FileGuard {
    fn open(path: &str) -> std::io::Result<Self> {
        Ok(FileGuard {
            file: std::fs::File::open(path)?,
        })
    }
}
```

### 2.4 策略模式

```rust
trait SortStrategy {
    fn sort(&self, data: &mut [i32]);
}

struct QuickSort;
struct MergeSort;

impl SortStrategy for QuickSort {
    fn sort(&self, data: &mut [i32]) { /* ... */ }
}

fn sort_with_strategy(data: &mut [i32], strategy: &dyn SortStrategy) {
    strategy.sort(data);
}
```

## 三、注意事项与常见陷阱

1. **所有权优先**：优先使用所有权而非引用计数
2. **trait 对象**：动态分发 vs 静态分发
3. **枚举优势**：用枚举代替空指针
4. **零成本抽象**：设计模式应是零成本的
5. **可读性**：模式应提高而非降低可读性
