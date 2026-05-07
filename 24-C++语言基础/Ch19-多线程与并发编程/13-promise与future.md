# promise与future

## 一、概念说明

`std::promise`和`std::future`用于线程间传递值。`promise`在写入端设置值，`future`在读取端获取值。`get()`只能调用一次，之后`future`失效。

```cpp
#include <iostream>
#include <thread>
#include <future>

void compute(std::promise<int> prom) {
    // 模拟计算
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    int result = 42;
    prom.set_value(result); // 设置结果
}

int main() {
    std::promise<int> prom;
    std::future<int> fut = prom.get_future();

    std::thread t(compute, std::move(prom));

    std::cout << "等待结果..." << std::endl;
    int value = fut.get(); // 阻塞等待
    std::cout << "结果: " << value << std::endl;

    t.join();
    return 0;
}
```

**输出：**
```
等待结果...
结果: 42
```

## 二、具体用法

### 2.1 传递异常

```cpp
#include <iostream>
#include <thread>
#include <future>
#include <stdexcept>

void mayFail(std::promise<int> prom) {
    try {
        // 模拟可能失败的操作
        throw std::runtime_error("计算出错");
        prom.set_value(42);
    } catch (...) {
        prom.set_exception(std::current_exception()); // 传递异常
    }
}

int main() {
    std::promise<int> prom;
    std::future<int> fut = prom.get_future();

    std::thread t(mayFail, std::move(prom));

    try {
        int val = fut.get(); // 重新抛出异常
        std::cout << "结果: " << val << std::endl;
    } catch (const std::exception& e) {
        std::cout << "捕获异常: " << e.what() << std::endl;
    }

    t.join();
    return 0;
}
```

**输出：**
```
捕获异常: 计算出错
```

### 2.2 shared_future

```cpp
#include <iostream>
#include <thread>
#include <future>

void notifier(std::promise<int> prom) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    prom.set_value(100);
}

void receiver(int id, std::shared_future<int> sf) {
    int val = sf.get(); // 多个线程可以共享同一个future
    std::cout << "接收者" << id << " 收到: " << val << std::endl;
}

int main() {
    std::promise<int> prom;
    std::shared_future<int> sf = prom.get_future().share();

    std::thread producer(notifier, std::move(prom));
    std::thread c1(receiver, 1, sf);
    std::thread c2(receiver, 2, sf);
    std::thread c3(receiver, 3, sf);

    producer.join(); c1.join(); c2.join(); c3.join();
    return 0;
}
```

**输出：**
```
接收者1 收到: 100
接收者2 收到: 100
接收者3 收到: 100
```

## 三、注意事项与常见陷阱

- **`future::get()`只能调用一次**：之后future变为无效状态。
- **`promise`只能设置一次值**：再次`set_value`会抛异常。
- **`set_exception`传递异常到future端**：`get()`会重新抛出。
- **`shared_future`可以多次`get()`**：支持多个消费者。
- **`promise`和`future`不可拷贝**：只能移动。
