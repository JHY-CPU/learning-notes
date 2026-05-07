# 智能指针与trait对象

## 一、概念说明

智能指针常与 trait 对象结合使用，实现动态分发和多态。`Box<dyn Trait>`、`Rc<dyn Trait>`、`Arc<dyn Trait>` 都是常见模式。

```rust
trait Animal {
    fn speak(&self) -> &str;
}

struct Dog;
struct Cat;

impl Animal for Dog { fn speak(&self) -> &str { "汪汪" } }
impl Animal for Cat { fn speak(&self) -> &str { "喵喵" } }

// 使用 Box<dyn Trait> 存储不同实现
let animals: Vec<Box<dyn Animal>> = vec![
    Box::new(Dog),
    Box::new(Cat),
];
```

## 二、具体用法

### 2.1 Box<dyn Trait> 单所有权

```rust
trait Processor {
    fn process(&self, data: &str) -> String;
}

struct UpperCase;
struct LowerCase;

impl Processor for UpperCase {
    fn process(&self, data: &str) -> String { data.to_uppercase() }
}

impl Processor for LowerCase {
    fn process(&self, data: &str) -> String { data.to_lowercase() }
}

fn create_processor(upper: bool) -> Box<dyn Processor> {
    if upper {
        Box::new(UpperCase)
    } else {
        Box::new(LowerCase)
    }
}
```

### 2.2 Rc<dyn Trait> 共享所有权

```rust
use std::rc::Rc;

trait Logger {
    fn log(&self, msg: &str);
}

struct ConsoleLogger;
struct FileLogger { path: String }

impl Logger for ConsoleLogger {
    fn log(&self, msg: &str) { println!("{}", msg); }
}

impl Logger for FileLogger {
    fn log(&self, msg: &str) {
        println!("写入文件 {}: {}", self.path, msg);
    }
}

// 多个组件共享同一个 logger
let logger: Rc<dyn Logger> = Rc::new(ConsoleLogger);
let logger2 = Rc::clone(&logger);
```

### 2.3 Arc<dyn Trait> 线程安全共享

```rust
use std::sync::Arc;

trait DataSource: Send + Sync {
    fn fetch(&self) -> Vec<u8>;
}

struct HttpSource { url: String }
struct FileSource { path: String }

impl DataSource for HttpSource {
    fn fetch(&self) -> Vec<u8> { vec![1, 2, 3] }
}

impl DataSource for FileSource {
    fn fetch(&self) -> Vec<u8> { vec![4, 5, 6] }
}

fn process_concurrent(source: Arc<dyn DataSource>) {
    use std::thread;
    let mut handles = vec![];
    for _ in 0..4 {
        let source = Arc::clone(&source);
        handles.push(thread::spawn(move || {
            source.fetch();
        }));
    }
}
```

## 三、注意事项与常见陷阱

1. **Send + Sync**：多线程 trait 对象需显式要求 Send + Sync
2. **虚函数开销**：动态分发有 vtable 查找开销
3. **大小不确定**：dyn Trait 大小不确定，必须在指针后使用
4. **trait bound**：与泛型相比，trait 对象更灵活但性能稍低
5. **object safety**：不是所有 trait 都能用作 trait 对象
