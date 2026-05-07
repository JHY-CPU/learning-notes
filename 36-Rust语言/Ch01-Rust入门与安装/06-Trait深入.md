# Trait 深入

## 一、概念说明

Trait 是 Rust 中定义共享行为的核心机制。深入理解 trait 有助于编写高质量的 Rust 代码。

```rust
// trait 定义
trait Summary {
    fn summarize(&self) -> String;

    // 默认实现
    fn preview(&self) -> String {
        format!("{}...", &self.summarize()[..20.min(self.summarize().len())])
    }
}

// 实现 trait
struct Article {
    title: String,
    content: String,
}

impl Summary for Article {
    fn summarize(&self) -> String {
        format!("{}: {}", self.title, &self.content[..50])
    }
}
```

## 二、Trait 作为参数

```rust
// impl Trait 语法
fn notify(item: &impl Summary) {
    println!("新闻: {}", item.summarize());
}

// trait bound 语法
fn notify<T: Summary>(item: &T) {
    println!("新闻: {}", item.summarize());
}

// 多个 trait bound
fn notify(item: &(impl Summary + Display)) {}

fn notify<T: Summary + Display>(item: &T) {}

// where 子句
fn some_function<T, U>(t: &T, u: &U) -> i32
where
    T: Display + Clone,
    U: Clone + Debug,
{
    // ...
    0
}
```

## 三、返回 impl Trait

```rust
// 返回实现了 trait 的类型
fn returns_summarizable() -> impl Summary {
    Article {
        title: String::from("标题"),
        content: String::from("内容"),
    }
}

// 注意：只能返回一种具体类型
fn returns_summarizable(switch: bool) -> impl Summary {
    if switch {
        Article { /* ... */ }
    } else {
        // Tweet { /* ... */ }  // 错误！不能返回不同类型
    }
}
```

## 四、Trait 对象

```rust
// trait 对象（动态分发）
fn draw_all(items: &Vec<Box<dyn Drawable>>) {
    for item in items {
        item.draw();
    }
}

// 或使用引用
fn draw_all(items: &[&dyn Drawable]) {
    for item in items {
        item.draw();
    }
}

// trait 对象的限制
// 1. 必须是对象安全的 trait
// 2. 方法不能返回 Self
// 3. 方法不能有泛型参数
```

## 五、关联类型

```rust
trait Iterator {
    type Item;  // 关联类型

    fn next(&mut self) -> Option<Self::Item>;
}

struct Counter {
    count: u32,
}

impl Iterator for Counter {
    type Item = u32;

    fn next(&mut self) -> Option<Self::Item> {
        self.count += 1;
        Some(self.count)
    }
}

// 关联类型 vs 泛型
// 关联类型：每个实现只能有一个
// 泛型：每个实现可以有多个
trait Container<T> {
    fn contains(&self, item: &T) -> bool;
}

// String 实现 Container<char>
impl Container<char> for String {
    fn contains(&self, item: &char) -> bool {
        self.contains(*item)
    }
}
```

## 六、注意事项

1. **孤儿规则**：至少 trait 或类型之一在当前 crate 中定义
2. **trait 对象开销**：动态分发有虚函数表开销
3. **静态分发优先**：impl Trait 比 dyn Trait 更高效
4. **对象安全**：不是所有 trait 都能用作 trait 对象
5. **覆盖默认实现**：可以覆盖 trait 的默认方法实现
