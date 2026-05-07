# placement new

## 一、概念说明

**placement new**允许在已分配的内存上直接构造对象，而不进行新的内存分配。它将对象的内存分配和构造分离开来，常用于**对象池**、**自定义内存管理器**和**性能关键**场景。

### 1.1 new的三种形式

- `new T`：标准new（分配内存+构造）
- `new(地址) T`：placement new（在指定地址构造）
- `operator new(大小)`：仅分配内存，不构造

## 二、具体用法

### 2.1 基本用法

```cpp
#include <iostream>
#include <new>  // placement new需要此头文件

class Widget {
    int value;
public:
    Widget(int v) : value(v) {
        std::cout << "构造: " << value << std::endl;
    }
    ~Widget() {
        std::cout << "析构: " << value << std::endl;
    }
    int getValue() const { return value; }
};

int main() {
    // 1. 预分配原始内存
    alignas(Widget) char buffer[sizeof(Widget)];

    // 2. 在已分配内存上构造对象
    Widget* w = new (buffer) Widget(42);

    std::cout << "值: " << w->getValue() << std::endl;

    // 3. 手动调用析构（不能delete，因为不是new分配的）
    w->~Widget();

    // 4. 可以在同一内存上重新构造
    Widget* w2 = new (buffer) Widget(100);
    std::cout << "新值: " << w2->getValue() << std::endl;
    w2->~Widget();

    return 0;
}
```

**输出：**
```
构造: 42
值: 42
析构: 42
构造: 100
新值: 100
析构: 100
```

### 2.2 对象池应用

```cpp
#include <iostream>
#include <vector>
#include <new>

class Pool {
    alignas(std::max_align_t) char memory[1024];
    size_t offset = 0;
public:
    void* allocate(size_t size) {
        if (offset + size > sizeof(memory))
            throw std::bad_alloc();
        void* ptr = memory + offset;
        offset += size;
        return ptr;
    }
};

class Connection {
    int id;
public:
    Connection(int id) : id(id) { std::cout << "连接" << id << "建立" << std::endl; }
    ~Connection() { std::cout << "连接" << id << "关闭" << std::endl; }
};

int main() {
    Pool pool;
    Connection* c1 = new (pool.allocate(sizeof(Connection))) Connection(1);
    Connection* c2 = new (pool.allocate(sizeof(Connection))) Connection(2);
    c1->~Connection();
    c2->~Connection();
    return 0;
}
```

**输出：**
```
连接1建立
连接2建立
连接1关闭
连接2关闭
```

## 三、注意事项与常见陷阱

- placement new构造的对象**不能用`delete`销毁**，必须手动调用析构函数
- 内存必须对齐到`alignof(T)`的要求
- 在同一内存上多次构造前，必须先析构之前对象
- placement new不分配内存，因此不会抛`std::bad_alloc`（构造函数仍可能抛异常）
- 需要`#include <new>`头文件
