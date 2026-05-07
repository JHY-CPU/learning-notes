# Deref与DerefMut

## 一、概念说明

`Deref` 和 `DerefMut` trait 允许自定义智能指针的解引用行为。实现这两个 trait 后，可以使用 `*` 运算符访问内部数据，也可以进行隐式类型转换。

```rust
use std::ops::{Deref, DerefMut};

struct MyBox<T>(T);

impl<T> Deref for MyBox<T> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.0
    }
}

let x = MyBox(5);
assert_eq!(5, *x); // 自动解引用
```

## 二、具体用法

### 2.1 自定义智能指针

```rust
use std::ops::{Deref, DerefMut};

struct SmartPtr<T> {
    data: T,
    name: String,
}

impl<T> SmartPtr<T> {
    fn new(data: T, name: &str) -> Self {
        SmartPtr { data, name: name.to_string() }
    }
}

impl<T> Deref for SmartPtr<T> {
    type Target = T;
    fn deref(&self) -> &T {
        println!("{} 被解引用", self.name);
        &self.data
    }
}

impl<T> DerefMut for SmartPtr<T> {
    fn deref_mut(&mut self) -> &mut T {
        println!("{} 被可变解引用", self.name);
        &mut self.data
    }
}
```

### 2.2 Deref 强制转换

```rust
fn hello(name: &str) {
    println!("你好，{}！", name);
}

let m = MyBox(String::from("Rust"));
// MyBox<String> → &String → &str
hello(&m); // 自动解引用转换
```

### 2.3 实现引用类型

```rust
use std::ops::Deref;

struct Wrapper<T> {
    inner: T,
}

impl<T> Wrapper<T> {
    fn new(inner: T) -> Self {
        Wrapper { inner }
    }
}

impl<T> Deref for Wrapper<T> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.inner
    }
}

let w = Wrapper::new(vec![1, 2, 3]);
// 可以直接调用 Vec 的方法
println!("长度: {}", w.len());
```

## 三、注意事项与常见陷阱

1. **不可变性**：Deref 只提供不可变引用
2. **隐式转换**：Deref 强制转换只适用于引用
3. **类型优先级**：Rust 优先使用显式解引用
4. **自动解引用**：方法调用时自动尝试多层解引用
5. **Clone 与 Deref**：Clone 和 Deref 是独立的概念
