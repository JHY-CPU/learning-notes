# unique_ptr详解

## 一、概念说明

`std::unique_ptr`是C++11引入的独占所有权智能指针。同一时刻只有一个`unique_ptr`拥有对象，销毁时自动释放资源。`unique_ptr`是零开销抽象，大小与裸指针相同。

## 二、具体用法

### 2.1 创建与使用

```cpp
#include <memory>

// 推荐方式
auto p = std::make_unique<int>(42);
std::cout << *p << std::endl;  // 输出: 42

// 数组形式（C++14）
auto arr = std::make_unique<int[]>(10);
arr[0] = 100;
std::cout << arr[0] << std::endl;  // 输出: 100

// 带构造参数
struct Point { int x, y; Point(int x, int y) : x(x), y(y) {} };
auto pt = std::make_unique<Point>(3, 4);
std::cout << pt->x << ", " << pt->y << std::endl;
// 输出: 3, 4
```

### 2.2 所有权转移

```cpp
auto p1 = std::make_unique<std::string>("Hello");

// 移动所有权（不可拷贝）
auto p2 = std::move(p1);

std::cout << (p1 == nullptr) << std::endl;  // 输出: 1
std::cout << *p2 << std::endl;              // 输出: Hello

// 函数间转移
void takeOwnership(std::unique_ptr<int> ptr) {
    std::cout << *ptr << std::endl;  // 输出: 42
}

auto p = std::make_unique<int>(42);
takeOwnership(std::move(p));  // p现在为空
```

### 2.3 自定义删除器

```cpp
// 文件句柄
auto fileDeleter = [](FILE* f) {
    if (f) {
        std::cout << "关闭文件\n";
        fclose(f);
    }
};

std::unique_ptr<FILE, decltype(fileDeleter)> file(
    fopen("test.txt", "r"), fileDeleter);

// 离开作用域时自动调用删除器
```

### 2.4 release 和 reset

```cpp
auto p = std::make_unique<int>(42);

// release：放弃所有权，返回裸指针
int* raw = p.release();
std::cout << (p == nullptr) << std::endl;  // 输出: 1
std::cout << *raw << std::endl;            // 输出: 42
delete raw;  // 需手动释放

// reset：释放当前对象，可替换为新对象
auto p2 = std::make_unique<int>(10);
p2.reset(new int(20));  // 释放10，持有20
p2.reset();              // 释放20，变为空
```

## 三、注意事项与常见陷阱

- `unique_ptr`不可拷贝，只能移动
- 不要用`make_unique`创建不完整类型（PIMPL模式中注意）
- 自定义删除器是类型的一部分，影响`unique_ptr`类型
- `release`后需要手动管理返回的裸指针
- `unique_ptr`大小通常等于裸指针（有删除器时可能更大）
