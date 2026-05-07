# mutex详解

## 一、概念说明

`std::mutex`是最基本的同步原语，提供独占访问。`lock()`获取锁，`unlock()`释放锁。同一时刻只有一个线程持有锁。

mutex类型：
- `mutex`：基本互斥量
- `recursive_mutex`：可重入，同一线程可多次lock
- `timed_mutex`：支持超时的lock
- `recursive_timed_mutex`：可重入+超时

```cpp
#include <iostream>
#include <thread>
#include <mutex>
#include <vector>

std::mutex mtx;
int sharedData = 0;

void safeAdd(int value) {
    mtx.lock();
    sharedData += value;
    std::cout << "线程 " << std::this_thread::get_id()
              << " 设置值: " << sharedData << std::endl;
    mtx.unlock(); // 必须手动unlock
}

int main() {
    std::vector<std::thread> threads;
    for (int i = 0; i < 5; ++i) {
        threads.emplace_back(safeAdd, (i + 1) * 10);
    }
    for (auto& t : threads) t.join();

    std::cout << "最终值: " << sharedData << std::endl;
    return 0;
}
```

**输出（顺序可能不同）：**
```
线程 140... 设置值: 10
线程 140... 设置值: 30
线程 140... 设置值: 60
线程 140... 设置值: 100
线程 140... 设置值: 150
最终值: 150
```

## 二、具体用法

### 2.1 try_lock和recursive_mutex

```cpp
#include <iostream>
#include <thread>
#include <mutex>

// try_lock: 非阻塞尝试获取锁
std::mutex mtx;

void tryLockDemo() {
    if (mtx.try_lock()) {
        std::cout << "获取锁成功" << std::endl;
        mtx.unlock();
    } else {
        std::cout << "锁已被占用" << std::endl;
    }
}

// recursive_mutex: 同一线程可重复加锁
std::recursive_mutex rmtx;

void recursiveFunc(int depth) {
    if (depth <= 0) return;
    std::lock_guard<std::recursive_mutex> lock(rmtx);
    std::cout << "递归深度: " << depth << std::endl;
    recursiveFunc(depth - 1); // 同一线程再次加锁
}

int main() {
    std::thread t1(tryLockDemo);
    std::thread t2(tryLockDemo);
    t1.join();
    t2.join();

    recursiveFunc(3);

    return 0;
}
```

**输出（顺序可能不同）：**
```
获取锁成功
锁已被占用
递归深度: 3
递归深度: 2
递归深度: 1
```

## 三、注意事项与常见陷阱

- **`lock()`后必须`unlock()`**：忘记会导致死锁，用`lock_guard`代替。
- **`try_lock()`不阻塞**：获取失败立即返回false。
- **`recursive_mutex`有性能开销**：尽量避免，重新设计代码。
- **mutex不可拷贝、不可移动**。
- **`timed_mutex`支持`try_lock_for`/`try_lock_until`**：超时后放弃。
