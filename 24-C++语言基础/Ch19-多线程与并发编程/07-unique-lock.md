# unique_lock

## 一、概念说明

`std::unique_lock`比`lock_guard`更灵活：支持延迟加锁、手动解锁、锁所有权转移。但有轻微的性能开销。它常与条件变量配合使用。

```cpp
#include <iostream>
#include <thread>
#include <mutex>

std::mutex mtx;

void demo() {
    std::unique_lock<std::mutex> lock(mtx, std::defer_lock); // 延迟加锁

    std::cout << "做一些非临界操作" << std::endl;

    lock.lock(); // 手动加锁
    std::cout << "临界区" << std::endl;
    lock.unlock(); // 手动解锁

    std::cout << "再做一些非临界操作" << std::endl;

    lock.lock(); // 再次加锁
    std::cout << "再次进入临界区" << std::endl;
} // 自动解锁

int main() {
    std::thread t1(demo);
    std::thread t2(demo);
    t1.join();
    t2.join();
    return 0;
}
```

**输出（顺序可能不同）：**
```
做一些非临界操作
临界区
再做一些非临界操作
再次进入临界区
做一些非临界操作
临界区
再做一些非临界操作
再次进入临界区
```

## 二、具体用法

### 2.1 锁所有权转移

```cpp
#include <iostream>
#include <thread>
#include <mutex>

std::mutex mtx;

std::unique_lock<std::mutex> getLock() {
    std::unique_lock<std::mutex> lock(mtx);
    std::cout << "函数内持有锁" << std::endl;
    return lock; // 移动语义转移所有权
}

int main() {
    auto lock = getLock(); // 获取锁所有权
    std::cout << "调用者持有锁" << std::endl;

    // 转移给条件变量
    // cv.wait(lock);

    return 0;
}
```

**输出：**
```
函数内持有锁
调用者持有锁
```

### 2.2 defer_lock / try_to_lock / adopt_lock

```cpp
#include <iostream>
#include <thread>
#include <mutex>

int main() {
    std::mutex mtx;

    // defer_lock: 不加锁
    std::unique_lock<std::mutex> l1(mtx, std::defer_lock);
    l1.lock(); // 之后手动加锁

    // try_to_lock: 尝试加锁（不阻塞）
    // std::unique_lock<std::mutex> l2(mtx, std::try_to_lock);
    // if (l2.owns_lock()) { /* 获取成功 */ }

    // adopt_lock: 假设已加锁
    mtx.lock();
    std::unique_lock<std::mutex> l3(mtx, std::adopt_lock);

    std::cout << "unique_lock三种策略演示" << std::endl;

    return 0;
}
```

**输出：**
```
unique_lock三种策略演示
```

## 三、注意事项与常见陷阱

- **`unique_lock`比`lock_guard`稍慢**：不需要灵活性时用`lock_guard`。
- **`owns_lock()`检查是否持有锁**：用于`try_to_lock`。
- **`release()`释放锁所有权**：不解锁，需手动unlock。
- **条件变量必须用`unique_lock`**：因为需要在`wait`中临时解锁。
- **移动语义**：`unique_lock`可以移动但不能拷贝。
