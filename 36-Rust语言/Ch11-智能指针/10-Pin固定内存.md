# Pin固定内存

## 一、概念说明

`Pin<T>` 用于确保值在内存中的位置不会被移动。这对于自引用结构和异步编程中 `Future` 的安全使用至关重要。

```rust
use std::pin::Pin;
use std::marker::PhantomPinned;

struct SelfReferential {
    data: String,
    ptr: *const String,
    _pin: PhantomPinned,
}

// 通过 Pin 确保结构体不会被移动
```

## 二、具体用法

### 2.1 基本概念

```rust
use std::pin::Pin;

// Pin 包装的值不能被移动
let pinned = Box::pin(42);
// pinned 不能被重新赋值或移动

// Pin 只影响是否可移动，不影响可变性
let mut pinned = Box::pin(vec![1, 2, 3]);
pinned.push(4); // 可以修改内部数据
```

### 2.2 自引用结构

```rust
use std::pin::Pin;
use std::marker::PhantomPinned;

struct SelfRef {
    data: String,
    slice: *const str,
    _pin: PhantomPinned,
}

impl SelfRef {
    fn new(data: String) -> Pin<Box<Self>> {
        let mut boxed = Box::pin(SelfRef {
            data,
            slice: std::ptr::null(),
            _pin: PhantomPinned,
        });

        let slice = &boxed.data as *const String as *const str;
        unsafe {
            let mut_ref = Pin::as_mut(&mut boxed);
            Pin::get_unchecked_mut(mut_ref).slice = slice;
        }

        boxed
    }

    fn get_slice(self: Pin<&Self>) -> &str {
        unsafe { &*self.slice }
    }
}
```

### 2.3 异步 Future

```rust
use std::pin::Pin;
use std::future::Future;
use std::task::{Context, Poll};

struct MyFuture {
    data: String,
}

impl Future for MyFuture {
    type Output = String;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        Poll::Ready(self.data.clone())
    }
}
```

## 三、注意事项与常见陷阱

1. **Unpin trait**：大多数类型实现 Unpin，可以安全移动
2. **堆分配**：Box::pin 创建在堆上的 pinned 值
3. **unsafe 使用**：Pin 的正确使用通常需要 unsafe 代码
4. **Drop 顺序**：pinned 值按声明顺序释放
5. **替代方案**：尽量避免自引用，使用索引代替指针
