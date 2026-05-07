# C++20 Coroutines

## 一、概念说明

C++20协程是无栈协程，使用三个关键字：
- `co_await`：挂起协程等待异步操作
- `co_yield`：暂停并返回值
- `co_return`：协程返回

协程由编译器生成状态机，比线程更轻量。

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
    for (int i = 0; i < 10 && gen.next(); ++i) {
        std::cout << gen.value() << " ";
    }
    std::cout << std::endl;
    return 0;
}
```

**输出：**
```
0 1 1 2 3 5 8 13 21 34
```

## 二、具体用法

### 2.1 promise_type接口

```cpp
// promise_type必须实现的方法：
// 1. get_return_object() - 返回协程的返回类型
// 2. initial_suspend()   - 协程开始时是否挂起
// 3. final_suspend()     - 协程结束时是否挂起
// 4. return_void() 或 return_value(T) - co_return的行为
// 5. yield_value(T)      - co_yield的行为（生成器需要）
// 6. unhandled_exception() - 异常处理

// suspend_always: 总是挂起
// suspend_never: 从不挂起
```

## 三、注意事项与常见陷阱

- **协程不是线程**：是协作式多任务，单线程执行。
- **`coroutine_handle`需要手动`destroy`**：否则内存泄漏。
- **协程的生命周期管理很复杂**：需要确保恢复前对象有效。
- **编译器支持不一致**：GCC 10+、Clang 12+、MSVC 19.28+。
- **C++23提供了`std::generator`**：无需自己实现。
