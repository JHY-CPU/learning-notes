# Drop资源释放

## 一、概念说明

`Drop` trait 允许自定义类型在离开作用域时的清理行为。Rust 编译器会自动在值离开作用域时调用 drop 方法。

```rust
struct CustomResource {
    name: String,
}

impl Drop for CustomResource {
    fn drop(&mut self) {
        println!("释放资源: {}", self.name);
    }
}

{
    let resource = CustomResource { name: "文件句柄".into() };
    // resource 在此处被使用
} // drop 被自动调用
```

## 二、具体用法

### 2.1 资源清理

```rust
struct FileHandle {
    path: String,
}

impl FileHandle {
    fn open(path: &str) -> Self {
        println!("打开文件: {}", path);
        FileHandle { path: path.to_string() }
    }
}

impl Drop for FileHandle {
    fn drop(&mut self) {
        println!("关闭文件: {}", self.path);
    }
}

fn process_file() {
    let file = FileHandle::open("data.txt");
    // 文件处理...
    // 离开作用域时自动关闭
}
```

### 2.2 手动 drop

```rust
let resource = CustomResource { name: "临时资源".into() };

// 使用 std::mem::drop 提前释放
std::mem::drop(resource);

// 或者显式调用 drop
// drop(resource);

// 此处 resource 不再可用
```

### 2.3 RAII 模式

```rust
struct Guard<T> {
    data: T,
    cleanup: Box<dyn Fn(&T)>,
}

impl<T> Guard<T> {
    fn new(data: T, cleanup: impl Fn(&T) + 'static) -> Self {
        Guard { data, cleanup: Box::new(cleanup) }
    }
}

impl<T> Drop for Guard<T> {
    fn drop(&mut self) {
        (self.cleanup)(&self.data);
        println!("资源已清理");
    }
}

let guard = Guard::new("数据库连接", |conn| {
    println!("关闭连接: {}", conn);
});
// 自动清理
```

## 三、注意事项与常见陷阱

1. **调用顺序**：变量按声明的逆序释放
2. **不能手动调用 drop**：drop 方法不能直接调用
3. **Drop 不能与 Copy 共存**：实现 Drop 的类型不能实现 Copy
4. **循环引用**：Drop 不能解决循环引用问题
5. **panic 安全**：drop 中的 panic 会导致程序中止
