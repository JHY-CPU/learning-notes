# coroutine进阶

## 一、概念说明

协程进阶涉及：自定义Awaiter、协程与异步IO集成、协程调度器、`generator`标准化（C++23）。理解`await_ready`/`await_suspend`/`await_resume`是自定义Awaiter的关键。

```cpp
#include <iostream>
#include <coroutine>
#include <chrono>
#include <thread>

// 自定义Awaiter：延迟执行
struct DelayAwaiter {
    std::chrono::milliseconds duration;

    bool await_ready() const { return duration.count() <= 0; }
    void await_suspend(std::coroutine_handle<>) {
        std::this_thread::sleep_for(duration);
    }
    void await_resume() const {}
};

// 使用co_await
struct VoidTask {
    struct promise_type {
        VoidTask get_return_object() {
            return VoidTask{std::coroutine_handle<promise_type>::from_promise(*this)};
        }
        std::suspend_never initial_suspend() { return {}; }
        std::suspend_always final_suspend() noexcept { return {}; }
        void return_void() {}
        void unhandled_exception() { std::terminate(); }
    };
    std::coroutine_handle<promise_type> handle;
    ~VoidTask() { if (handle) handle.destroy(); }
};

VoidTask delayedHello() {
    std::cout << "开始..." << std::endl;
    co_await DelayAwaiter{std::chrono::milliseconds(100)};
    std::cout << "延迟100ms后完成" << std::endl;
}

int main() {
    auto task = delayedHello();
    task.handle.resume();
    return 0;
}
```

**输出：**
```
开始...
延迟100ms后完成
```

## 二、具体用法

### 2.1 co_await的三个方法

```cpp
// Awaiter接口：
// bool await_ready();
//   返回true表示结果已就绪，不需要挂起
//   返回false表示需要挂起
//
// void/H auto await_suspend(std::coroutine_handle<> h);
//   挂起时调用，可在这里注册回调或启动异步操作
//   返回void：挂起协程
//   返回bool：true挂起，false不挂起
//   返回coroutine_handle：跳转到另一个协程
//
// T await_resume();
//   恢复时调用，返回co_await表达式的值

// 示例：简单Awaiter
struct AlwaysReady {
    bool await_ready() { return true; }
    void await_suspend(std::coroutine_handle<>) {}
    int await_resume() { return 42; }
};

// 示例：条件Awaiter
struct ConditionalAwaiter {
    bool shouldSuspend;
    bool await_ready() { return !shouldSuspend; }
    void await_suspend(std::coroutine_handle<>) {
        // 可以在这里启动异步操作
    }
    void await_resume() {}
};
```

### 2.2 C++23 std::generator

```cpp
// C++23提供了标准的generator
// #include <generator>
// std::generator<int> fibonacci() {
//     int a = 0, b = 1;
//     while (true) {
//         co_yield a;
//         auto temp = a;
//         a = b;
//         b = temp + b;
//     }
// }

// 使用
// for (int fib : fibonacci()) {
//     if (fib > 100) break;
//     std::cout << fib << " ";
// }
```

## 三、注意事项与常见陷阱

- **`await_suspend`中不要抛异常**：协程已经挂起。
- **协程的生命周期管理**：`coroutine_handle`销毁需调用`destroy()`。
- **对称传输（symmetric transfer）**：`await_suspend`返回另一个协程句柄实现跳转。
- **协程不是线程**：单线程中协程是协作式多任务。
- **协程与线程池结合**：异步操作完成后在池中恢复协程。
