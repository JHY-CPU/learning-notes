# Weak弱引用

## 一、概念说明

`Weak<T>` 是 `Rc<T>` 或 `Arc<T>` 的弱引用版本。弱引用不增加强引用计数，可以被升级为强引用。用于打破循环引用导致的内存泄漏。

```rust
use std::rc::{Rc, Weak};

let strong = Rc::new(5);
let weak = Rc::downgrade(&strong);

// 升级为强引用
if let Some(upgraded) = weak.upgrade() {
    println!("值: {}", *upgraded);
}

// 强引用被释放后，升级返回 None
drop(strong);
assert!(weak.upgrade().is_none());
```

## 二、具体用法

### 2.1 父子关系

```rust
use std::rc::{Rc, Weak};
use std::cell::RefCell;

struct Parent {
    children: Vec<Rc<RefCell<Child>>>,
}

struct Child {
    parent: Weak<RefCell<Parent>>,
    name: String,
}

fn create_family() {
    let parent = Rc::new(RefCell::new(Parent { children: vec![] }));
    let child = Rc::new(RefCell::new(Child {
        parent: Rc::downgrade(&parent),
        name: "孩子".into(),
    }));

    parent.borrow_mut().children.push(child);
    // 不会造成循环引用
}
```

### 2.2 缓存观察者

```rust
use std::rc::{Rc, Weak};
use std::cell::RefCell;

struct Cache {
    observers: RefCell<Vec<Weak<dyn Observer>>>,
}

trait Observer {
    fn notify(&self, data: &str);
}

impl Cache {
    fn add_observer(&self, observer: Weak<dyn Observer>) {
        self.observers.borrow_mut().push(observer);
    }

    fn notify_all(&self, data: &str) {
        let mut observers = self.observers.borrow_mut();
        observers.retain(|weak| {
            if let Some(observer) = weak.upgrade() {
                observer.notify(data);
                true
            } else {
                false // 已被释放，移除
            }
        });
    }
}
```

### 2.3 查询引用计数

```rust
use std::rc::Rc;

let strong = Rc::new(vec![1, 2, 3]);
let weak1 = Rc::downgrade(&strong);
let weak2 = Rc::downgrade(&strong);

println!("强引用: {}", Rc::strong_count(&strong)); // 1
println!("弱引用: {}", Rc::weak_count(&strong));   // 2
```

## 三、注意事项与常见陷阱

1. **生命周期**：Weak 不影响数据的释放
2. **升级检查**：upgrade 返回 Option，需处理 None 情况
3. **循环引用**：使用 Weak 打破父子结构的循环引用
4. **性能开销**：Weak 也有引用计数开销
5. **线程安全**：Arc 对应的 Weak 是线程安全的
