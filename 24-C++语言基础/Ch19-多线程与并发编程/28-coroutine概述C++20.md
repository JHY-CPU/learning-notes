# coroutine概述 C++20

## 一、概念说明

C++20协程是一种可以暂停和恢复执行的函数，使用`co_await`、`co_yield`、`co_return`关键字。协程避免了回调地狱，用同步写法实现异步逻辑。

协程的三类返回类型：`generator`（生成器）、`task`（异步任务）、`lazy`（延迟计算）。

```cpp
#include <iostream>
#include <coroutine>
#include <optional>

// 简单的Generator协程
template <typename T>
struct Generator {
    struct promise_type {
        T currentValue;
        Generator get_return_object() {
            return Generator{std::coroutine_handle<promise_type>::from_promise(*this)};
        }
        std::suspend_always initial_suspend() { return {}; }
        std::suspend_always final_suspend() noexcept { return {}; }
        std::suspend_always yield_value(T value) {
            currentValue = value;
            return {};
        }
        void return_void() {}
        void unhandled_exception() { std::terminate(); }
    };

    std::coroutine_handle<promise_type> handle;

    explicit Generator(std::coroutine_handle<promise_type> h) : handle(h) {}
    ~Generator() { if (handle) handle.destroy(); }

    bool next() { handle.resume(); return !handle.done(); }
    T value() const { return handle.promise().currentValue; }
};

Generator<int> range(int start, int end) {
    for (int i = start; i < end; ++i) {
        co_yield i; // 暂停并产生值
    }
}

int main() {
    auto gen = range(1, 5);
    while (gen.next()) {
        std::cout << gen.value() << " ";
    }
    std::cout << std::endl;
    return 0;
}
```

**输出：**
```
1 2 3 4
```

## 二、具体用法

### 2.1 协程关键字

| 关键字 | 作用 |
|--------|------|
| `co_await` | 挂起协程，等待异步操作 |
| `co_yield` | 暂停并返回值（生成器） |
| `co_return` | 协程返回，设置最终值 |

```cpp
#include <iostream>
#include <coroutine>

// co_return示例
struct Task {
    struct promise_type {
        int result;
        Task get_return_object() {
            return Task{std::coroutine_handle<promise_type>::from_promise(*this)};
        }
        std::suspend_never initial_suspend() { return {}; }
        std::suspend_always final_suspend() noexcept { return {}; }
        void return_value(int val) { result = val; }
        void unhandled_exception() { std::terminate(); }
    };

    std::coroutine_handle<promise_type> handle;
    ~Task() { if (handle) handle.destroy(); }
    int get() { return handle.promise().result; }
};

Task compute() {
    co_return 42;
}

int main() {
    auto task = compute();
    std::cout << "结果: " << task.get() << std::endl;
    return 0;
}
```

**输出：**
```
结果: 42
```

## 三、注意事项与常见陷阱

- **C++20协程是无栈协程**：由编译器生成状态机。
- **`promise_type`必须实现特定接口**：`get_return_object`、`initial_suspend`等。
- **协程句柄需要手动`destroy`**：避免内存泄漏。
- **`co_await`/`co_yield`/`co_return`只能在协程中使用**。
- **编译器支持不一致**：GCC 10+、Clang 12+、MSVC 2019 16.8+。
