# Fn/FnMut/FnOnce三大trait

## 一、概念说明

Rust 闭包有三种 trait：`Fn`、`FnMut`、`FnOnce`。它们定义了闭包如何捕获和使用环境变量，以及闭包可以被调用的次数。

```rust
// Fn: 不可变借用环境，可多次调用
// FnMut: 可变借用环境，可多次调用
// FnOnce: 获取所有权，只能调用一次
```

## 二、具体用法

### 2.1 Fn 不可变借用

```rust
let name = String::from("Rust");

// Fn trait: 通过 &self 捕获
let greet = || {
    println!("你好，{}！", name);
    // name 被不可变借用
};

greet();
greet(); // 可以多次调用

// 函数参数约束
fn call_twice<F: Fn()>(f: F) {
    f();
    f();
}

call_twice(greet);
```

### 2.2 FnMut 可变借用

```rust
let mut count = 0;

// FnMut trait: 通过 &mut self 捕获
let mut increment = || {
    count += 1; // 需要可变借用
    println!("计数: {}", count);
};

increment(); // 1
increment(); // 2

// 函数参数约束
fn call_three_times<F: FnMut()>(mut f: F) {
    f();
    f();
    f();
}

let mut total = 0;
call_three_times(|| {
    total += 1;
});
```

### 2.3 FnOnce 所有权转移

```rust
let data = vec![1, 2, 3, 4, 5];

// FnOnce trait: 通过 self 捕获，消耗所有权
let consume = || {
    let sum: i32 = data.iter().sum();
    println!("总和: {}", sum);
    drop(data); // 显式释放
};

consume();
// consume(); // 编译错误：data 已被消耗

// 函数参数约束
fn run_once<F: FnOnce()>(f: F) {
    f();
}

let message = String::from("只有一次");
run_once(move || {
    println!("{}", message);
    // message 被消耗
});
```

### 2.4 trait 层级关系

```rust
// Fn 继承 FnMut 继承 FnOnce
// 所有 Fn 闭包同时也是 FnMut 和 FnOnce
// 所有 FnMut 闭包同时也是 FnOnce

fn accept_fn_once<F: FnOnce()>(f: F) { f(); }
fn accept_fn_mut<F: FnMut()>(mut f: F) { f(); }
fn accept_fn<F: Fn()>(f: F) { f(); }

// Fn 闭包可以传递给任何要求
let x = 5;
let closure = || println!("{}", x);
accept_fn_once(closure);  // OK
accept_fn_mut(closure);   // OK
accept_fn(closure);       // OK
```

## 三、注意事项与常见陷阱

1. **自动选择**：编译器根据捕获方式自动推断最合适的 trait
2. **move 关键字**：move 不一定意味着 FnOnce，取决于捕获变量的使用方式
3. **Copy 类型**：Copy 类型 move 后原变量仍可用
4. **trait bound**：选择最宽松的满足需求的 trait bound
5. **Box<dyn Fn()>**：需要动态分发时使用 trait 对象
