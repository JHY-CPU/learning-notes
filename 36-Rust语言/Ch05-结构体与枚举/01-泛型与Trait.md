# 泛型与 Trait

## 一、概念说明

泛型（Generics）允许编写适用于多种类型的代码。Trait 定义共享的行为，类似其他语言的接口。

```rust
// 泛型函数
fn largest<T: PartialOrd>(list: &[T]) -> &T {
    let mut largest = &list[0];
    for item in &list[1..] {
        if item > largest {
            largest = item;
        }
    }
    largest
}

// Trait 定义
trait Summary {
    fn summarize(&self) -> String;

    // 默认实现
    fn preview(&self) -> String {
        format!("{}...", &self.summarize()[..20])
    }
}
```

## 二、泛型详解

```rust
// 泛型结构体
struct Point<T> {
    x: T,
    y: T,
}

// 多类型泛型
struct MixedPoint<T, U> {
    x: T,
    y: U,
}

// 泛型方法
impl<T: std::fmt::Display> Point<T> {
    fn display(&self) {
        println!("({}, {})", self.x, self.y);
    }
}

// 为特定类型实现方法
impl Point<f64> {
    fn distance_from_origin(&self) -> f64 {
        (self.x.powi(2) + self.y.powi(2)).sqrt()
    }
}

// 泛型枚举
enum Result<T, E> {
    Ok(T),
    Err(E),
}

fn main() {
    let int_point = Point { x: 5, y: 10 };
    let float_point = Point { x: 1.0, y: 4.0 };

    int_point.display();
    float_point.display();
    println!("距离: {}", float_point.distance_from_origin());
}
```

## 三、Trait 详解

```rust
// 定义 trait
trait Drawable {
    fn draw(&self);

    // 带默认实现的方法
    fn bounding_box(&self) -> (f64, f64, f64, f64) {
        (0.0, 0.0, 0.0, 0.0)
    }
}

// 实现 trait
struct Circle {
    radius: f64,
}

impl Drawable for Circle {
    fn draw(&self) {
        println!("绘制半径为 {} 的圆", self.radius);
    }

    fn bounding_box(&self) -> (f64, f64, f64, f64) {
        let r = self.radius;
        (-r, -r, r * 2.0, r * 2.0)
    }
}

struct Rectangle {
    width: f64,
    height: f64,
}

impl Drawable for Rectangle {
    fn draw(&self) {
        println!("绘制 {} x {} 的矩形", self.width, self.height);
    }
}

// trait 作为参数
fn draw_shape(shape: &impl Drawable) {
    shape.draw();
}

// trait bound 语法
fn draw_shape_v2<T: Drawable>(shape: &T) {
    shape.draw();
}

// 多个 trait bound
fn draw_and_print<T: Drawable + std::fmt::Debug>(shape: &T) {
    shape.draw();
    println!("{:?}", shape);
}

// where 子句（复杂约束）
fn complex_function<T, U>(t: &T, u: &U) -> String
where
    T: Drawable + Clone,
    U: Drawable + std::fmt::Debug,
{
    t.draw();
    format!("{:?}", u)
}
```

## 四、常见标准库 Trait

```rust
// Debug - 调试输出
#[derive(Debug)]
struct Person {
    name: String,
    age: u32,
}

// Clone - 深拷贝
#[derive(Clone)]
struct Config {
    debug: bool,
}

// PartialEq - 相等比较
#[derive(PartialEq)]
enum Direction {
    North,
    South,
    East,
    West,
}

// Display - 用户友好的输出
impl std::fmt::Display for Person {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{} ({}岁)", self.name, self.age)
    }
}

fn main() {
    let person = Person { name: String::from("张三"), age: 25 };
    println!("{:?}", person);   // Debug 输出
    println!("{}", person);     // Display 输出

    let config = Config { debug: true };
    let config2 = config.clone();

    assert!(Direction::North == Direction::North);
}
```

## 五、注意事项

1. **单态化**：泛型在编译时生成具体类型的代码，零运行时开销
2. **孤儿规则**：只能为当前 crate 的类型实现 trait
3. **trait 对象**：使用 `dyn Trait` 实现动态分发
4. **关联类型**：trait 中可以定义关联类型
5. **生命周期**：trait 方法可能需要生命周期标注
