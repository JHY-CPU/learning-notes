# C++20 Coroutines

## 一、概念说明

C++20协程（C++20 §8.6, §17.12）是无栈协程（Stackless Coroutines），使用三个关键字控制：`co_await`（挂起等待）、`co_yield`（暂停返回值）、`co_return`（协程返回）。协程由编译器生成状态机，比线程更轻量，适合异步I/O、惰性生成器等场景。

### 1.1 协程 vs 线程

| 特性 | 协程 | 线程 |
|------|------|------|
| 调度 | 协作式（主动让出） | 抢占式（OS调度） |
| 栈 | 无栈（堆上状态机） | 有栈（独立栈空间） |
| 开销 | 极小（仅状态机） | 较大（栈+上下文切换） |
| 通信 | 直接返回值 | 需要同步机制 |

```cpp
#include <iostream>
#include <coroutine>

// 简单的生成器
template <typename T>
struct Generator {
    struct promise_type {
        T current;
        Generator get_return_object() {
            return Generator{std::coroutine_handle<promise_type>::from_promise(*this)};
        }
        std::suspend_always initial_suspend() { return {}; }
        std::suspend_always final_suspend() noexcept { return {}; }
        std::suspend_always yield_value(T val) {
            current = val;
            return {};
        }
        void return_void() {}
        void unhandled_exception() { std::terminate(); }
    };

    std::coroutine_handle<promise_type> handle;

    explicit Generator(std::coroutine_handle<promise_type> h) : handle(h) {}
    ~Generator() { if (handle) handle.destroy(); }

    // 禁止拷贝
    Generator(const Generator&) = delete;
    Generator& operator=(const Generator&) = delete;

    // 允许移动
    Generator(Generator&& other) noexcept : handle(other.handle) {
        other.handle = nullptr;
    }

    bool next() { handle.resume(); return !handle.done(); }
    T value() const { return handle.promise().current; }
};

Generator<int> fibonacci() {
    int a = 0, b = 1;
    while (true) {
        co_yield a;
        auto tmp = b;
        b = a + b;
        a = tmp;
    }
}

int main() {
    auto gen = fibonacci();
    std::cout << "Fibonacci: ";
    for (int i = 0; i < 10 && gen.next(); ++i) {
        std::cout << gen.value() << " ";
    }
    std::cout << std::endl;
    return 0;
}
```

**输出：**
```
Fibonacci: 0 1 1 2 3 5 8 13 21 34
```

## 二、具体用法

### 2.1 promise_type接口

协程的`promise_type`必须实现以下方法：

```cpp
// promise_type必须实现的方法：
struct promise_type {
    // 1. get_return_object() - 返回协程的返回类型
    Coroutine get_return_object();

    // 2. initial_suspend() - 协程开始时是否挂起
    //    suspend_always: 惰性（调用者手动resume）
    //    suspend_never: 立即开始执行
    std::suspend_always initial_suspend();

    // 3. final_suspend() - 协程结束时是否挂起
    std::suspend_always final_suspend() noexcept;

    // 4. return_void() 或 return_value(T) - co_return的行为
    void return_void();
    // 或
    void return_value(T val);

    // 5. yield_value(T) - co_yield的行为（生成器需要）
    std::suspend_always yield_value(T val);

    // 6. unhandled_exception() - 异常处理
    void unhandled_exception() { std::terminate(); }
};
```

### 2.2 co_await的使用

```cpp
#include <iostream>
#include <coroutine>
#include <thread>
#include <chrono>

// 简单的任务
struct Task {
    struct promise_type {
        Task get_return_object() {
            return Task{std::coroutine_handle<promise_type>::from_promise(*this)};
        }
        std::suspend_never initial_suspend() { return {}; }
        std::suspend_always final_suspend() noexcept { return {}; }
        void return_void() {}
        void unhandled_exception() { std::terminate(); }
    };

    std::coroutine_handle<promise_type> handle;
    ~Task() { if (handle) handle.destroy(); }
};

// 简单的awaiter
struct SleepAwaiter {
    int ms;
    bool await_ready() const { return ms <= 0; }
    void await_suspend(std::coroutine_handle<>) {
        std::this_thread::sleep_for(std::chrono::milliseconds(ms));
    }
    void await_resume() {}
};

Task asyncWork() {
    std::cout << "开始工作" << std::endl;
    co_await SleepAwaiter{100}; // 模拟异步等待
    std::cout << "工作完成" << std::endl;
}
```

### 2.3 C++23 std::generator

C++23提供了标准的`std::generator`，无需自己实现。

```cpp
// C++23（概念性）
// #include <generator>
// std::generator<int> range(int start, int end) {
//     for (int i = start; i < end; ++i)
//         co_yield i;
// }
// for (int x : range(0, 10)) std::cout << x << " ";
```

## 三、注意事项与常见陷阱

1. **协程不是线程**：是协作式多任务，单线程执行，主动`co_await`时让出控制权。
2. **`coroutine_handle`需要手动`destroy`**：否则内存泄漏。
3. **协程的生命周期管理很复杂**：需要确保恢复前对象有效。
4. **编译器支持不一致**：GCC 10+、Clang 12+、MSVC 19.28+。
5. **C++23提供了`std::generator`**：无需自己实现生成器。
6. **协程不适合CPU密集型任务**：适合I/O密集型、惰性生成等场景。
