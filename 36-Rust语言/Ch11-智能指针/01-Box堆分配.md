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

## 三、注意事项与常见陷阱

1. **所有权**：Box 是唯一所有者，不能共享
2. **自动解引用**：Box 实现了 Deref，可以直接调用内部类型的方法
3. **内存泄漏**：循环引用的 Box 会导致内存泄漏
4. **Clone**：Box 的 clone 会深拷贝堆上的数据
5. **性能**：堆分配有开销，小数据不需要 Box
