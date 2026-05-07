# Cell基本类型内部可变

## 一、概念说明

`Cell<T>` 提供了针对 `Copy` 类型的内部可变性。与 RefCell 不同，Cell 不提供引用，而是通过 get/set 方法操作值，因此没有借用检查开销。

```rust
use std::cell::Cell;

let cell = Cell::new(5);
cell.set(10);
println!("{}", cell.get()); // 10

// 交换值
let cell2 = Cell::new(20);
cell.swap(&cell2);
println!("cell: {}, cell2: {}", cell.get(), cell2.get());
```

## 二、具体用法

### 2.1 Copy 类型操作

```rust
use std::cell::Cell;

// 整数
let counter = Cell::new(0);
counter.set(counter.get() + 1);

// 浮点数
let value = Cell::new(3.14);
value.set(value.get() * 2.0);

// 布尔
let flag = Cell::new(true);
flag.set(!flag.get());

// 数组（固定大小）
let arr = Cell::new([1, 2, 3]);
let mut inner = arr.get();
inner[0] = 10;
arr.set(inner);
```

### 2.2 结构体中的 Cell

```rust
use std::cell::Cell;

struct Counter {
    count: Cell<u64>,
    name: String,
}

impl Counter {
    fn new(name: &str) -> Self {
        Counter {
            count: Cell::new(0),
            name: name.to_string(),
        }
    }

    fn increment(&self) {
        // 不可变方法中修改计数
        self.count.set(self.count.get() + 1);
    }

    fn get_count(&self) -> u64 {
        self.count.get()
    }
}

let counter = Counter::new("访问计数器");
counter.increment();
counter.increment();
println!("{}: {}", counter.name, counter.get_count());
```

### 2.3 Cell 与 RefCell 对比

```rust
use std::cell::{Cell, RefCell};

// Cell: 适用于 Copy 类型，无运行时检查
let cell = Cell::new(5);
cell.set(10);
let val = cell.get();

// RefCell: 适用于任意类型，有运行时借用检查
let refcell = RefCell::new(vec![1, 2, 3]);
refcell.borrow_mut().push(4);
let borrowed = refcell.borrow();
```

## 三、注意事项与常见陷阱

1. **Copy 约束**：Cell 只适用于实现 Copy 的类型
2. **无借用检查**：Cell 没有借用概念，直接获取/设置值
3. **性能**：Cell 比 RefCell 更轻量，没有运行时借用检查
4. **线程安全**：Cell 不是线程安全的
5. **适用场景**：简单的计数器、标志位等使用 Cell 最合适
