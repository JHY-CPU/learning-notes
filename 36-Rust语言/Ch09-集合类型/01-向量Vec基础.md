# 向量Vec基础

## 一、概念说明

`Vec<T>` 是 Rust 标准库中最常用的集合类型，称为**向量（Vector）**。它是一个可增长的、堆分配的数组，可以在运行时动态调整大小。与数组不同，向量的大小不需要在编译时确定。

```rust
// 创建空向量
let v: Vec<i32> = Vec::new();

// 使用宏创建并初始化
let v = vec![1, 2, 3, 4, 5];

// 使用 with_capacity 预分配空间
let v: Vec<i32> = Vec::with_capacity(10);
```

## 二、具体用法

### 2.1 添加和删除元素

```rust
let mut v = Vec::new();

// push 在末尾添加元素
v.push(10);
v.push(20);
v.push(30);

// pop 移除并返回最后一个元素
let last = v.pop(); // Some(30)

// insert 在指定位置插入
v.insert(1, 15); // [10, 15, 20]

// remove 移除指定位置元素
let removed = v.remove(0); // 10
```

### 2.2 访问元素

```rust
let v = vec![10, 20, 30, 40, 50];

// 索引访问（越界会 panic）
let third: &i32 = &v[2]; // 30

// get 方法访问（返回 Option）
match v.get(2) {
    Some(val) => println!("第三个元素: {}", val),
    None => println!("没有第三个元素"),
}
```

### 2.3 遍历向量

```rust
let v = vec![100, 32, 57];

// 不可变引用遍历
for i in &v {
    println!("{}", i);
}

// 可变引用遍历（修改元素）
let mut v = vec![100, 32, 57];
for i in &mut v {
    *i += 50;
}
```

### 2.4 使用枚举存储不同类型

```rust
enum SpreadsheetCell {
    Int(i32),
    Float(f64),
    Text(String),
}

let row = vec![
    SpreadsheetCell::Int(3),
    SpreadsheetCell::Text(String::from("蓝色")),
    SpreadsheetCell::Float(10.12),
];
```

## 三、注意事项与常见陷阱

1. **借用规则**：不能在同一作用域内同时持有可变和不可变引用
2. **内存释放**：向量离开作用域时，其所有元素都会被释放
3. **容量管理**：频繁 push 时使用 `with_capacity` 可减少重新分配次数
4. **越界访问**：使用 `[]` 索引越界会 panic，建议使用 `get` 方法
5. **切片引用**：`&v[..]` 可将向量转为切片使用
