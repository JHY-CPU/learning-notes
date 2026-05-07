# RefCell内部可变性

## 一、概念说明

`RefCell<T>` 提供了内部可变性模式，允许在拥有不可变引用时修改内部数据。借用检查从编译时推迟到运行时。

```rust
use std::cell::RefCell;

let data = RefCell::new(5);

// 不可变借用
let borrow1 = data.borrow();
println!("{}", *borrow1);

// 可变借用
drop(borrow1);
let mut borrow2 = data.borrow_mut();
*borrow2 = 10;
```

## 二、具体用法

### 2.1 运行时借用检查

```rust
use std::cell::RefCell;

let data = RefCell::new(vec![1, 2, 3]);

// 同时可有多个不可变借用
let borrow1 = data.borrow();
let borrow2 = data.borrow();
println!("{:?} {:?}", *borrow1, *borrow2);

// 释放不可变借用后才能获取可变借用
drop(borrow1);
drop(borrow2);

let mut borrow_mut = data.borrow_mut();
borrow_mut.push(4);
```

### 2.2 配合 Rc 使用

```rust
use std::rc::Rc;
use std::cell::RefCell;

#[derive(Debug)]
struct SharedState {
    count: i32,
    messages: Vec<String>,
}

let shared = Rc::new(RefCell::new(SharedState {
    count: 0,
    messages: vec![],
}));

let clone1 = Rc::clone(&shared);
let clone2 = Rc::clone(&shared);

clone1.borrow_mut().count += 1;
clone2.borrow_mut().messages.push("新消息".into());

println!("{:?}", shared.borrow());
```

### 2.3 Mock 测试

```rust
use std::cell::RefCell;

struct Logger {
    messages: RefCell<Vec<String>>,
}

impl Logger {
    fn new() -> Self {
        Logger { messages: RefCell::new(vec![]) }
    }

    fn log(&self, msg: &str) {
        // 不可变方法中修改内部状态
        self.messages.borrow_mut().push(msg.to_string());
    }

    fn get_messages(&self) -> Vec<String> {
        self.messages.borrow().clone()
    }
}
```

## 三、注意事项与常见陷阱

1. **运行时开销**：运行时借用检查有性能开销
2. **panic 风险**：违反借用规则会导致 panic
3. **单线程**：RefCell 不是 Send/Sync，只能在单线程使用
4. **借用嵌套**：注意避免在持有借用时获取另一个借用
5. **替代方案**：简单场景考虑 Cell（仅适用于 Copy 类型）
