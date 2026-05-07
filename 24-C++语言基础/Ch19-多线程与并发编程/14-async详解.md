# async详解

## 一、概念说明

`std::async`是最高级的异步编程接口，启动一个异步任务并返回`future`。它自动管理线程的创建和同步。

启动策略：
- `std::launch::async`：强制新线程执行
- `std::launch::deferred`：延迟到`get()`时同步执行
- 默认：由实现决定

```cpp
#include <iostream>
#include <future>
#include <chrono>

int computeHeavy(int n) {
    std::cout << "计算中..." << std::endl;
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    return n * n;
}

int main() {
    // 启动异步任务
    auto fut = std::async(std::launch::async, computeHeavy, 10);

    std::cout << "主线程继续工作" << std::endl;
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // 获取结果（阻塞直到完成）
    int result = fut.get();
    std::cout << "结果: " << result << std::endl;

    return 0;
}
```

**输出：**
```
计算中...
主线程继续工作
结果: 100
```

## 二、具体用法

### 2.1 并行计算

```cpp
#include <iostream>
#include <future>
#include <vector>

long long sumRange(int start, int end) {
    long long sum = 0;
    for (int i = start; i < end; ++i) sum += i;
    return sum;
}

int main() {
    int n = 1000000;
    int mid = n / 2;

    // 并行计算两半
    auto fut1 = std::async(std::launch::async, sumRange, 0, mid);
    auto fut2 = std::async(std::launch::async, sumRange, mid, n);

    long long total = fut1.get() + fut2.get();
    std::cout << "0到" << n-1 << "的和: " << total << std::endl;

    return 0;
}
```

**输出：**
```
0到999999的和: 499999500000
```

### 2.2 超时和等待

```cpp
#include <iostream>
#include <future>
#include <chrono>

int slowTask() {
    std::this_thread::sleep_for(std::chrono::seconds(2));
    return 42;
}

int main() {
    auto fut = std::async(std::launch::async, slowTask);

    // 等待最多500ms
    if (fut.wait_for(std::chrono::milliseconds(500)) == std::future_status::ready) {
        std::cout << "任务完成: " << fut.get() << std::endl;
    } else {
        std::cout << "任务还在运行，继续等待..." << std::endl;
        std::cout << "最终结果: " << fut.get() << std::endl;
    }

    return 0;
}
```

**输出：**
```
任务还在运行，继续等待...
最终结果: 42
```

## 三、注意事项与常见陷阱

- **`async`返回的`future`析构时会阻塞**：确保保存future或调用`get()`。
- **默认策略可能同步执行**：需要并发时显式指定`launch::async`。
- **`async`不是线程池**：每次调用可能创建新线程。
- **不能捕获`async`的返回值**：临时`future`析构会阻塞。
- **C++17/20中`async`的前景不明**：推荐使用`std::jthread`或线程池。
