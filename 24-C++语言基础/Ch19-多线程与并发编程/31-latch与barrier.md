# latch与barrier C++20

## 一、概念说明

C++20引入了两种同步原语（`<latch>`头文件）：
- **`std::latch`**：一次性屏障，计数归零后所有等待线程释放。不可重用。
- **`std::barrier`**：可重用屏障，所有线程到达后一起继续。支持完成函数。

```cpp
#include <iostream>
#include <thread>
#include <latch>
#include <vector>

int main() {
    const int numWorkers = 4;
    std::latch latch(numWorkers);

    std::vector<std::thread> workers;
    for (int i = 0; i < numWorkers; ++i) {
        workers.emplace_back([i, &latch] {
            std::cout << "工作者" << i << " 完成初始化" << std::endl;
            latch.count_down(); // 计数减1
            // 继续做其他工作...
        });
    }

    std::cout << "主线程等待所有工作者就绪..." << std::endl;
    latch.wait(); // 等待计数归零
    std::cout << "所有工作者已就绪！" << std::endl;

    for (auto& t : workers) t.join();
    return 0;
}
```

**输出（顺序可能不同）：**
```
主线程等待所有工作者就绪...
工作者0 完成初始化
工作者1 完成初始化
工作者2 完成初始化
工作者3 完成初始化
所有工作者已就绪！
```

## 二、具体用法

### 2.1 barrier可重用屏障

```cpp
#include <iostream>
#include <thread>
#include <barrier>
#include <vector>

int main() {
    const int numThreads = 3;
    const int numRounds = 3;

    // 完成函数：每轮所有线程到达后调用
    auto onComplete = [] {
        std::cout << "--- 一轮结束 ---" << std::endl;
    };

    std::barrier barrier(numThreads, onComplete);

    auto worker = [&](int id) {
        for (int round = 0; round < numRounds; ++round) {
            std::cout << "线程" << id << " 第" << round << "轮工作" << std::endl;
            barrier.arrive_and_wait(); // 到达并等待
        }
    };

    std::vector<std::thread> threads;
    for (int i = 0; i < numThreads; ++i) {
        threads.emplace_back(worker, i);
    }
    for (auto& t : threads) t.join();

    return 0;
}
```

**输出（顺序可能不同）：**
```
线程0 第0轮工作
线程1 第0轮工作
线程2 第0轮工作
--- 一轮结束 ---
线程0 第1轮工作
...
```

### 2.2 latch vs barrier

| 特性 | `latch` | `barrier` |
|------|---------|-----------|
| 可重用 | 否 | 是 |
| 完成函数 | 无 | 有 |
| 用途 | 一次性同步 | 多轮迭代同步 |
| 灵活性 | 简单 | 更灵活 |

```cpp
// latch: 适合等待N个任务完成
std::latch done(3);
// ... 3个线程各执行 count_down() ...
done.wait(); // 一次性等待

// barrier: 适合迭代算法
std::barrier sync(3);
for (int iter = 0; iter < 10; ++iter) {
    // ... 每个线程做一轮计算 ...
    sync.arrive_and_wait(); // 每轮同步
}
```

## 三、注意事项与常见陷阱

- **`latch`不可重用**：计数归零后不能再用。
- **`barrier::arrive_and_wait`等价于`arrive()`+`wait()`**。
- **`barrier`的完成函数在最后一个到达的线程中执行**。
- **`arrive()`可以与`wait()`分开使用**：先到达做其他事，再等待。
- **`latch`和`barrier`需要C++20**：GCC 11+、Clang 14+。
