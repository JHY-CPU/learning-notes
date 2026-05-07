# std::thread基础

## 一、概念说明

`std::thread`是C++11引入的线程类，创建线程时指定可调用对象（函数、lambda、函数对象）和参数。线程创建后立即开始执行，必须在析构前调用`join()`（等待结束）或`detach()`（分离）。

```cpp
#include <iostream>
#include <thread>
#include <string>

// 普通函数
void hello() {
    std::cout << "Hello from thread " << std::this_thread::get_id() << std::endl;
}

// 带参数的函数
void greet(const std::string& name, int times) {
    for (int i = 0; i < times; ++i) {
        std::cout << name << " says hi #" << i << std::endl;
    }
}

int main() {
    // 1. 函数
    std::thread t1(hello);

    // 2. Lambda
    std::thread t2([]() {
        std::cout << "Lambda thread" << std::endl;
    });

    // 3. 带参数（参数被拷贝到线程中）
    std::string name = "Alice";
    std::thread t3(greet, name, 3);

    t1.join();
    t2.join();
    t3.join();

    return 0;
}
```

**输出（顺序可能不同）：**
```
Hello from thread 140234567890
Lambda thread
Alice says hi #0
Alice says hi #1
Alice says hi #2
```

## 二、具体用法

### 2.1 参数传递

```cpp
#include <iostream>
#include <thread>
#include <string>

void byValue(int x) {
    std::cout << "值传递: " << x << std::endl;
}

void byRef(int& x) {
    x += 10;
    std::cout << "引用传递: " << x << std::endl;
}

int main() {
    int val = 42;

    // 值传递（默认）
    std::thread t1(byValue, val);

    // 引用传递：必须用std::ref
    std::thread t2(byRef, std::ref(val));

    t1.join();
    t2.join();

    std::cout << "主线程val: " << val << std::endl; // 52
    return 0;
}
```

**输出：**
```
值传递: 42
引用传递: 52
主线程val: 52
```

### 2.2 成员函数作为线程入口

```cpp
#include <iostream>
#include <thread>

class Worker {
    int id;
public:
    Worker(int i) : id(i) {}
    void doWork(int n) {
        for (int i = 0; i < n; ++i) {
            std::cout << "Worker " << id << " step " << i << std::endl;
        }
    }
};

int main() {
    Worker w(1);
    // 成员函数：传对象指针和参数
    std::thread t(&Worker::doWork, &w, 3);
    t.join();

    return 0;
}
```

**输出：**
```
Worker 1 step 0
Worker 1 step 1
Worker 1 step 2
```

## 三、注意事项与常见陷阱

- **`join()`后线程不可再`join`**：用`joinable()`检查。
- **`detach()`后不能再`join`**：线程变为后台运行。
- **参数默认拷贝**：引用需用`std::ref()`包装。
- **移动语义**：`thread`不可拷贝，只能移动。
- **未`join`或`detach`的线程析构时调用`std::terminate()`**。
