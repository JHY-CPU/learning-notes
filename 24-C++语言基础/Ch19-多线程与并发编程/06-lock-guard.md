# lock_guard

## 一、概念说明

`std::lock_guard`是RAII锁守卫，构造时加锁、析构时自动解锁。不需要手动`unlock`，即使发生异常也能正确释放锁。

```cpp
#include <iostream>
#include <thread>
#include <mutex>
#include <vector>

std::mutex mtx;
int counter = 0;

void increment(int times) {
    for (int i = 0; i < times; ++i) {
        std::lock_guard<std::mutex> lock(mtx); // 自动加锁
        ++counter;
    } // 自动解锁
}

int main() {
    std::vector<std::thread> threads;
    for (int i = 0; i < 4; ++i) {
        threads.emplace_back(increment, 100000);
    }
    for (auto& t : threads) t.join();

    std::cout << "counter = " << counter << std::endl;
    return 0;
}
```

**输出：**
```
counter = 400000
```

## 二、具体用法

### 2.1 异常安全

```cpp
#include <iostream>
#include <thread>
#include <mutex>

std::mutex mtx;

void riskyOperation() {
    std::lock_guard<std::mutex> lock(mtx);
    std::cout << "临界区开始" << std::endl;

    // 即使抛异常，lock_guard析构也会释放锁
    // throw std::runtime_error("错误");

    std::cout << "临界区结束" << std::endl;
} // 自动unlock

int main() {
    std::thread t1(riskyOperation);
    std::thread t2(riskyOperation);
    t1.join();
    t2.join();
    return 0;
}
```

**输出：**
```
临界区开始
临界区结束
临界区开始
临界区结束
```

### 2.2 adopt_lock

```cpp
#include <iostream>
#include <thread>
#include <mutex>

std::mutex mtx1, mtx2;

void transfer() {
    // 先手动加锁
    mtx1.lock();
    mtx2.lock();

    // 用adopt_lock告诉lock_guard锁已持有
    std::lock_guard<std::mutex> lock1(mtx1, std::adopt_lock);
    std::lock_guard<std::mutex> lock2(mtx2, std::adopt_lock);

    std::cout << "两个锁都持有" << std::endl;

    // 析构时释放，但不重复加锁
}

int main() {
    std::thread t(transfer);
    t.join();
    return 0;
}
```

**输出：**
```
两个锁都持有
```

## 三、注意事项与常见陷阱

- **`lock_guard`不能手动解锁**：作用域结束才释放，用`unique_lock`代替。
- **`lock_guard`不能拷贝和移动**。
- **`adopt_lock`假设锁已持有**：如果未加锁则未定义行为。
- **`lock_guard`只管理一个mutex**：多个锁用`scoped_lock`（C++17）。
- **锁的粒度应尽可能小**：只保护需要同步的最小代码段。
