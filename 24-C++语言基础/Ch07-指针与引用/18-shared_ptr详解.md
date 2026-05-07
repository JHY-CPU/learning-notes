# shared_ptr详解

## 一、概念说明

`std::shared_ptr`实现**共享所有权**语义，多个`shared_ptr`可以指向同一对象。通过**引用计数**追踪所有者数量，最后一个`shared_ptr`销毁时释放对象。

## 二、具体用法

### 2.1 基本使用

```cpp
auto sp1 = std::make_shared<int>(42);
std::cout << "计数: " << sp1.use_count() << std::endl;  // 输出: 计数: 1

{
    auto sp2 = sp1;  // 共享所有权
    std::cout << "计数: " << sp1.use_count() << std::endl;  // 输出: 计数: 2
}
// sp2销毁
std::cout << "计数: " << sp1.use_count() << std::endl;  // 输出: 计数: 1
```

### 2.2 make_shared vs new

```cpp
// 推荐：make_shared（一次内存分配）
auto sp1 = std::make_shared<std::string>("Hello");

// 不推荐：两次分配（对象 + 控制块）
std::shared_ptr<std::string> sp2(new std::string("Hello"));
```

### 2.3 共享容器

```cpp
std::vector<std::shared_ptr<int>> vec;
auto val = std::make_shared<int>(42);

vec.push_back(val);  // 共享所有权
std::cout << "计数: " << val.use_count() << std::endl;  // 输出: 计数: 2

vec.clear();  // vec中的shared_ptr销毁
std::cout << "计数: " << val.use_count() << std::endl;  // 输出: 计数: 1
```

### 2.4 自定义删除器

```cpp
auto deleter = [](int* p) {
    std::cout << "释放: " << *p << std::endl;
    delete p;
};

std::shared_ptr<int> sp(new int(42), deleter);
sp.reset();
// 输出: 释放: 42
```

### 2.5 从unique_ptr转换

```cpp
auto up = std::make_unique<int>(42);
std::shared_ptr<int> sp = std::move(up);  // unique_ptr → shared_ptr
// up现在为空
std::cout << *sp << std::endl;  // 输出: 42
// 反向转换不可行
```

### 2.6 enable_shared_from_this

```cpp
struct Widget : std::enable_shared_from_this<Widget> {
    std::shared_ptr<Widget> getPtr() {
        return shared_from_this();  // 安全获取自身的shared_ptr
    }
};

auto w = std::make_shared<Widget>();
auto w2 = w->getPtr();
std::cout << w.use_count() << std::endl;  // 输出: 2
```

## 三、注意事项与常见陷阱

- `make_shared`比`new`更高效（单次分配）和安全（异常安全）
- 不要用裸指针创建多个独立的`shared_ptr`，会导致双重释放
- `shared_ptr`引用计数有原子操作开销
- 循环引用导致内存泄漏，用`weak_ptr`打破
- `shared_ptr`的控制块与对象可能不一起释放（`make_shared`时）
