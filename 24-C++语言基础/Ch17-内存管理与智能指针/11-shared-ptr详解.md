# shared_ptr详解

## 一、概念说明

`std::shared_ptr`通过**引用计数**实现共享所有权。多个`shared_ptr`可以指向同一对象，当最后一个`shared_ptr`销毁时，对象被释放。

`std::make_shared<T>(args...)`是推荐的创建方式，它只需一次内存分配（对象和控制块一起分配），比`shared_ptr(new T())`更高效。

```cpp
#include <iostream>
#include <memory>

class Connection {
    std::string name;
public:
    Connection(std::string n) : name(std::move(n)) {
        std::cout << "Connection '" << name << "' 建立" << std::endl;
    }
    ~Connection() {
        std::cout << "Connection '" << name << "' 关闭" << std::endl;
    }
    void query() const { std::cout << "查询: " << name << std::endl; }
};

int main() {
    std::shared_ptr<Connection> p1 = std::make_shared<Connection>("MySQL");
    std::cout << "引用计数: " << p1.use_count() << std::endl;

    {
        auto p2 = p1; // 拷贝，引用计数+1
        std::cout << "引用计数: " << p1.use_count() << std::endl;
        p2->query();
    } // p2销毁，引用计数-1

    std::cout << "引用计数: " << p1.use_count() << std::endl;
    return 0;
}
```

**输出：**
```
Connection 'MySQL' 建立
引用计数: 1
引用计数: 2
查询: MySQL
引用计数: 1
Connection 'MySQL' 关闭
```

## 二、具体用法

### 2.1 make_shared vs shared_ptr(new)

```cpp
#include <iostream>
#include <memory>

struct Data {
    int value;
    Data(int v) : value(v) { std::cout << "Data(" << value << ") 构造" << std::endl; }
    ~Data() { std::cout << "Data(" << value << ") 析构" << std::endl; }
};

int main() {
    // make_shared: 一次内存分配（推荐）
    auto p1 = std::make_shared<Data>(42);

    // shared_ptr(new): 两次内存分配（不推荐）
    std::shared_ptr<Data> p2(new Data(99));

    std::cout << "p1引用计数: " << p1.use_count() << std::endl;
    std::cout << "p2引用计数: " << p2.use_count() << std::endl;

    return 0;
}
```

**输出：**
```
Data(42) 构造
Data(99) 构造
p1引用计数: 1
p2引用计数: 1
Data(99) 析构
Data(42) 析构
```

### 2.2 别名构造

```cpp
#include <iostream>
#include <memory>

struct Parent {
    int id;
    Parent(int i) : id(i) { std::cout << "Parent(" << id << ")" << std::endl; }
    ~Parent() { std::cout << "~Parent(" << id << ")" << std::endl; }
};

int main() {
    auto parent = std::make_shared<Parent>(1);

    // 别名构造：共享parent的控制块，但指向parent的成员
    std::shared_ptr<int> alias(parent, &parent->id);

    std::cout << "parent引用计数: " << parent.use_count() << std::endl;
    std::cout << "alias引用计数: " << alias.use_count() << std::endl;
    std::cout << "alias指向的值: " << *alias << std::endl;

    return 0;
}
```

**输出：**
```
Parent(1)
parent引用计数: 2
alias引用计数: 2
alias指向的值: 1
~Parent(1)
```

## 三、注意事项与常见陷阱

- **`make_shared`只需一次分配**：比`shared_ptr(new T())`更高效。
- **循环引用会导致内存泄漏**：用`weak_ptr`打破循环。
- **不要用同一裸指针创建多个`shared_ptr`**：每个会创建独立的控制块，导致重复释放。
- **引用计数操作是原子的**：线程安全但对象本身的访问不是。
- **`use_count()`在多线程中不准确**：仅用于调试。
