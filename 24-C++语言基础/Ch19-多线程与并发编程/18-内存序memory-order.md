# 内存序 memory_order

## 一、概念说明

内存序定义了原子操作周围的内存访问如何排序。C++定义了6种内存序，从最宽松（`relaxed`）到最严格（`seq_cst`）。它们控制编译器和CPU的重排序行为。

- `memory_order_relaxed`：只保证原子性，不保证顺序
- `memory_order_acquire`：当前读操作之后的读写不能重排到前面
- `memory_order_release`：当前写操作之前的读写不能重排到后面
- `memory_order_acq_rel`：同时具有acquire和release语义
- `memory_order_seq_cst`：完全顺序一致性（默认）
- `memory_order_consume`：数据依赖顺序（几乎不用）

```cpp
#include <iostream>
#include <thread>
#include <atomic>

std::atomic<bool> ready{false};
int data = 0;

void producer() {
    data = 42;                                          // 1
    ready.store(true, std::memory_order_release);       // 2
    // release保证1在2之前对其他线程可见
}

void consumer() {
    while (!ready.load(std::memory_order_acquire)) {}   // 3
    // acquire保证3之后可以看到1的写入
    std::cout << "data = " << data << std::endl;        // 4
}

int main() {
    std::thread t1(producer);
    std::thread t2(consumer);
    t1.join();
    t2.join();
    return 0;
}
```

**输出：**
```
data = 42
```

## 二、具体用法

### 2.1 relaxed顺序

```cpp
#include <iostream>
#include <thread>
#include <atomic>

std::atomic<int> x{0}, y{0};
int r1 = 0, r2 = 0;

void thread1() {
    x.store(1, std::memory_order_relaxed);
    r1 = y.load(std::memory_order_relaxed);
}

void thread2() {
    y.store(1, std::memory_order_relaxed);
    r2 = x.load(std::memory_order_relaxed);
}

int main() {
    // relaxed允许各种重排，可能输出 r1=0, r2=0
    // （尽管x和y都设为1了）
    // 在seq_cst下不会出现这种情况

    std::cout << "relaxed顺序不保证全局一致性" << std::endl;
    std::cout << "可能结果: r1=0, r2=0（在seq_cst下不可能）" << std::endl;

    return 0;
}
```

**输出：**
```
relaxed顺序不保证全局一致性
可能结果: r1=0, r2=0（在seq_cst下不可能）
```

### 2.2 内存序对比

| 内存序 | 开销 | 语义 |
|--------|------|------|
| `relaxed` | 最低 | 只原子性 |
| `consume` | 低 | 数据依赖 |
| `acquire` | 中 | 读屏障 |
| `release` | 中 | 写屏障 |
| `acq_rel` | 较高 | 读写屏障 |
| `seq_cst` | 最高 | 全序一致 |

## 三、注意事项与常见陷阱

- **默认`seq_cst`最安全**：除非你完全理解其他内存序的语义。
- **`relaxed`适合简单计数器**：不需要同步其他数据。
- **`acquire`/`release`是最常用的非默认内存序**：实现发布-订阅模式。
- **`consume`目前几乎等价于`acquire`**：编译器未实现数据依赖优化。
- **x86是强内存模型**：`acquire`/`release`/`seq_cst`编译结果相同。
- **ARM/PowerPC是弱内存模型**：内存序的影响更明显。
