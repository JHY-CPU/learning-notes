# Box堆分配

## 一、概念说明

`Box<T>` 是 Rust 中最简单的智能指针，用于在堆上分配数据。Box 拥有其指向的数据，当 Box 离开作用域时，堆上的数据也会被释放。

```rust
// 在堆上分配一个整数
let b = Box::new(5);
println!("b = {}", b);

// Box 用于递归类型
enum List {
    Cons(i32, Box<List>),
    Nil,
}
```

## 二、具体用法

### 2.1 基本用法

```rust
// 堆上分配大数据
let large_data = Box::new([0u8; 1024 * 1024]); // 1MB

// 减少栈帧大小
fn process_large() {
    let data = Box::new(vec![0; 1_000_000]);
    // data 在堆上，不占用栈空间
}

// 解引用
let x = 5;
let y = Box::new(x);
assert_eq!(5, *y);
```

### 2.2 递归类型

```rust
// 链表
enum List<T> {
    Cons(T, Box<List<T>>),
    Nil,
}

impl<T> List<T> {
    fn new() -> Self {
        List::Nil
    }

    fn prepend(self, value: T) -> Self {
        List::Cons(value, Box::new(self))
    }
}

let list = List::new()
    .prepend(3)
    .prepend(2)
    .prepend(1);
```

### 2.3 trait 对象

```rust
trait Draw {
    fn draw(&self);
}

struct Circle { radius: f64 }
struct Square { side: f64 }

impl Draw for Circle {
    fn draw(&self) { println!("绘制圆形 r={}", self.radius); }
}

impl Draw for Square {
    fn draw(&self) { println!("绘制方形 s={}", self.side); }
}

// 存储不同类型的实现
let shapes: Vec<Box<dyn Draw>> = vec![
    Box::new(Circle { radius: 5.0 }),
    Box::new(Square { side: 3.0 }),
];

for shape in &shapes {
    shape.draw();
}
```

### 2.4 Box 的内存布局与 Pin

```rust
use std::pin::Pin;

// Pin 确保数据不会被移动，适用于自引用结构
fn pin_example() {
    // Box 本身在栈上，数据在堆上
    let boxed = Box::new(42);
    // 栈: [ptr] -> 堆: [42]

    // Pin<Box<T>> 确保 T 不会被移动
    let pinned: Pin<Box<String>> = Box::pin(String::from("不能移动我"));
    // pinned.as_ref() 获取 &T
    // pinned.as_mut() 获取 Pin<&mut T>
}
```

### 2.5 Box 与内存对齐

```rust
fn alignment_example() {
    // 堆分配保证正确的内存对齐
    let aligned: Box<[u8; 64]> = Box::new([0; 64]);

    // 对于非常大的数据，Box 可以避免栈溢出
    let huge: Box<[u8; 10 * 1024 * 1024]> = Box::new([0; 10 * 1024 * 1024]);
    // 如果不用 Box，10MB 数据在栈上会导致栈溢出

    // Box 自动对齐到类型要求的边界
    let aligned_u64: Box<u64> = Box::new(42);
    // 对齐到 8 字节边界
}
```

### 2.6 Box::leak 泄漏内存

```rust
fn leak_example() {
    let boxed = Box::new(String::from("泄漏的字符串"));

    // Box::leak 将 Box 转换为 &'static mut T
    // 数据永远不会被释放
    let leaked: &'static mut String = Box::leak(boxed);

    // 使用场景：需要 'static 生命周期的全局配置
    leaked.push_str(" - 已修改");

    // 注意：这会导致内存泄漏，仅在必要时使用
}
```

### 2.7 Box 和 dyn trait 的分发

```rust
trait Shape {
    fn area(&self) -> f64;
    fn name(&self) -> &str;
}

struct Circle(f64);
struct Rectangle(f64, f64);

impl Shape for Circle {
    fn area(&self) -> f64 { std::f64::consts::PI * self.0 * self.0 }
    fn name(&self) -> &str { "圆形" }
}

impl Shape for Rectangle {
    fn area(&self) -> f64 { self.0 * self.1 }
    fn name(&self) -> &str { "矩形" }
}

fn dynamic_dispatch_example() {
    // 动态分发：运行时确定调用哪个实现
    let shapes: Vec<Box<dyn Shape>> = vec![
        Box::new(Circle(5.0)),
        Box::new(Rectangle(3.0, 4.0)),
    ];

    for shape in &shapes {
        // 通过 vtable 调用方法
        println!("{}: 面积 = {:.2}", shape.name(), shape.area());
    }

    // 静态分发对比：编译时确定，更快
    fn print_shape<S: Shape>(shape: &S) {
        println!("{}: 面积 = {:.2}", shape.name(), shape.area());
    }
}
```

## 四、实际应用场景

### 4.1 配置管理

```rust
use std::collections::HashMap;

struct AppConfig {
    settings: Box<HashMap<String, String>>, // 大配置避免栈溢出
}

impl AppConfig {
    fn new() -> Self {
        AppConfig {
            settings: Box::new(HashMap::new()),
        }
    }

    fn get(&self, key: &str) -> Option<&str> {
        self.settings.get(key).map(|s| s.as_str())
    }
}
```

### 4.2 递归数据结构

```rust
// 二叉搜索树
enum BST<T: Ord> {
    Empty,
    Node {
        value: T,
        left: Box<BST<T>>,
        right: Box<BST<T>>,
    },
}

impl<T: Ord> BST<T> {
    fn new() -> Self {
        BST::Empty
    }

    fn insert(&mut self, value: T) {
        match self {
            BST::Empty => {
                *self = BST::Node {
                    value,
                    left: Box::new(BST::Empty),
                    right: Box::new(BST::Empty),
                };
            }
            BST::Node { value: ref mut v, left, right } => {
                if value < *v {
                    left.insert(value);
                } else {
                    right.insert(value);
                }
            }
        }
    }

    fn contains(&self, value: &T) -> bool {
        match self {
            BST::Empty => false,
            BST::Node { value: v, left, right } => {
                if value == v {
                    true
                } else if value < v {
                    left.contains(value)
                } else {
                    right.contains(value)
                }
            }
        }
    }
}
```

### 4.3 Error trait 对象

```rust
fn parse_and_process() -> Result<i32, Box<dyn std::error::Error>> {
    let content = std::fs::read_to_string("input.txt")?;
    let number: i32 = content.trim().parse()?;
    Ok(number * 2)
    // 可以返回任何实现了 Error 的错误类型
}
```

## 五、性能考虑

1. **堆分配开销**：每次 `Box::new` 都有一次 malloc 调用（约 30-100ns）
2. **缓存局部性**：堆数据不连续，可能导致缓存未命中
3. **编译优化**：Release 模式下编译器会内联小 Box
4. **分配器选择**：可使用 `allocator_api` 特性指定自定义分配器

## 六、注意事项与常见陷阱

1. **所有权**：Box 是唯一所有者，不能共享，需要共享使用 `Rc` 或 `Arc`
2. **自动解引用**：Box 实现了 Deref，可以直接调用内部类型的方法，支持 `*` 和 `.` 操作
3. **内存泄漏**：循环引用的 Box 会导致内存泄漏，需要手动打破循环
4. **Clone**：Box 的 clone 会深拷贝堆上的数据，可能很昂贵
5. **性能**：堆分配有开销，小数据不需要 Box，优先使用栈分配
6. **Sized 约束**：`Box<T>` 要求 T 是 Sized，`Box<dyn Trait>` 例外
7. **Drop 顺序**：Box drop 时会递归 drop 内部数据
