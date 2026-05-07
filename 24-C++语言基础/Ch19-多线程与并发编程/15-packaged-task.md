# packaged_task

## 一、概念说明

`std::packaged_task`将可调用对象（函数、lambda、函数对象）包装成一个可异步获取结果的任务。调用`operator()`执行任务，通过关联的`future`获取返回值。

```cpp
#include <iostream>
#include <future>
#include <thread>

int multiply(int a, int b) {
    return a * b;
}

int main() {
    // 包装函数
    std::packaged_task<int(int, int)> task(multiply);

    // 获取future
    std::future<int> fut = task.get_future();

    // 在线程中执行
    std::thread t(std::move(task), 6, 7);

    // 获取结果
    std::cout << "6 * 7 = " << fut.get() << std::endl;

    t.join();
    return 0;
}
```

**输出：**
```
6 * 7 = 42
```

## 二、具体用法

### 2.1 任务队列

```cpp
#include <iostream>
#include <future>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>

class TaskQueue {
    std::queue<std::packaged_task<void()>> tasks;
    std::mutex mtx;
    std::condition_variable cv;
    bool stop_ = false;

public:
    template <typename F, typename... Args>
    auto enqueue(F&& f, Args&&... args)
        -> std::future<typename std::result_of<F(Args...)>::type> {
        using ReturnType = typename std::result_of<F(Args...)>::type;

        auto task = std::make_shared<std::packaged_task<ReturnType()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...)
        );

        std::future<ReturnType> fut = task->get_future();
        {
            std::lock_guard<std::mutex> lock(mtx);
            tasks.emplace([task](){ (*task)(); });
        }
        cv.notify_one();
        return fut;
    }

    void worker() {
        while (true) {
            std::packaged_task<void()> task;
            {
                std::unique_lock<std::mutex> lock(mtx);
                cv.wait(lock, [this]{ return !tasks.empty() || stop_; });
                if (stop_ && tasks.empty()) return;
                task = std::move(tasks.front());
                tasks.pop();
            }
            task();
        }
    }

    void stop() {
        {
            std::lock_guard<std::mutex> lock(mtx);
            stop_ = true;
        }
        cv.notify_all();
    }
};

int main() {
    TaskQueue queue;
    std::thread worker(&TaskQueue::worker, &queue);

    auto fut1 = queue.enqueue([](){ return 1 + 1; });
    auto fut2 = queue.enqueue([](){ return 2 * 3; });

    std::cout << "1+1 = " << fut1.get() << std::endl;
    std::cout << "2*3 = " << fut2.get() << std::endl;

    queue.stop();
    worker.join();
    return 0;
}
```

**输出：**
```
1+1 = 2
2*3 = 6
```

### 2.2 与async对比

| 特性 | `async` | `packaged_task` |
|------|---------|----------------|
| 线程管理 | 自动 | 手动 |
| 执行策略 | 可选 | 自定义 |
| 任务队列 | 不支持 | 支持 |
| 复杂度 | 简单 | 灵活 |

## 三、注意事项与常见陷阱

- **`packaged_task`不可拷贝**：只能移动。
- **`get_future`只能调用一次**。
- **`packaged_task`调用`operator()`后不能再次调用**。
- **`valid()`检查是否仍可调用**。
- **异常会传递到future**：`get()`会重新抛出。
