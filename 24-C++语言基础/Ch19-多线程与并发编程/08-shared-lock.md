# shared_lock（读写锁）

## 一、概念说明

`std::shared_lock`（C++14）配合`std::shared_mutex`（C++17）实现读写锁：多个线程可以同时读（共享锁），但写操作需要独占锁。

- `shared_mutex::lock_shared()` / `shared_lock`：共享（读）锁
- `shared_mutex::lock()` / `unique_lock`：独占（写）锁

```cpp
#include <iostream>
#include <thread>
#include <shared_mutex>
#include <vector>

std::shared_mutex smtx;
int sharedValue = 0;

// 多个读线程可以同时执行
void reader(int id) {
    std::shared_lock<std::shared_mutex> lock(smtx);
    std::cout << "读者" << id << " 读取值: " << sharedValue << std::endl;
}

// 写操作独占
void writer(int id, int value) {
    std::unique_lock<std::shared_mutex> lock(smtx);
    sharedValue = value;
    std::cout << "写者" << id << " 写入值: " << value << std::endl;
}

int main() {
    std::vector<std::thread> threads;

    // 启动多个读者（可以并发）
    for (int i = 0; i < 3; ++i) {
        threads.emplace_back(reader, i);
    }

    // 启动一个写者
    threads.emplace_back(writer, 1, 42);

    // 再启动读者
    for (int i = 3; i < 5; ++i) {
        threads.emplace_back(reader, i);
    }

    for (auto& t : threads) t.join();
    return 0;
}
```

**输出（顺序可能不同）：**
```
读者0 读取值: 0
读者1 读取值: 0
读者2 读取值: 0
写者1 写入值: 42
读者3 读取值: 42
读者4 读取值: 42
```

## 二、具体用法

### 2.1 缓存读写分离

```cpp
#include <iostream>
#include <thread>
#include <shared_mutex>
#include <unordered_map>
#include <string>

class ThreadSafeCache {
    std::unordered_map<std::string, int> data;
    mutable std::shared_mutex mtx;

public:
    // 读操作：共享锁
    int get(const std::string& key) const {
        std::shared_lock<std::shared_mutex> lock(mtx);
        auto it = data.find(key);
        return it != data.end() ? it->second : -1;
    }

    // 写操作：独占锁
    void set(const std::string& key, int value) {
        std::unique_lock<std::shared_mutex> lock(mtx);
        data[key] = value;
    }

    size_t size() const {
        std::shared_lock<std::shared_mutex> lock(mtx);
        return data.size();
    }
};

int main() {
    ThreadSafeCache cache;

    std::thread w([&cache]() { cache.set("key1", 100); });
    w.join();

    std::thread r1([&cache]() { std::cout << "key1=" << cache.get("key1") << std::endl; });
    std::thread r2([&cache]() { std::cout << "size=" << cache.size() << std::endl; });
    r1.join();
    r2.join();

    return 0;
}
```

**输出：**
```
key1=100
size=1
```

## 三、注意事项与常见陷阱

- **`shared_mutex`在C++17中引入**：`shared_lock`在C++14中引入。
- **读多写少的场景收益最大**：频繁写入时与普通mutex差别不大。
- **`shared_lock`不能升级为独占锁**：需要先释放再获取独占锁。
- **`std::shared_timed_mutex`支持超时**：C++14引入。
- **避免写饥饿**：某些实现中写者可能长时间等待。
