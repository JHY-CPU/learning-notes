# placement new

## 一、概念说明

Placement new允许在**已分配的内存区域**上调用构造函数，不分配新内存。常用于内存池、对象重用、自定义内存管理等场景。

语法：`new (address) Type(args...)`

```cpp
#include <iostream>
#include <new>

class Widget {
    int id;
public:
    Widget(int i) : id(i) {
        std::cout << "构造 Widget(" << id << ")" << std::endl;
    }
    ~Widget() {
        std::cout << "析构 Widget(" << id << ")" << std::endl;
    }
    void show() const { std::cout << "Widget ID: " << id << std::endl; }
};

int main() {
    // 分配原始内存
    alignas(Widget) char buffer[sizeof(Widget)];

    // 在buffer上构造对象（不分配新内存）
    Widget* w = new (buffer) Widget(42);
    w->show();

    // 手动调用析构（不能用delete，因为buffer不在堆上）
    w->~Widget();

    // 可以在同一内存上再次构造
    Widget* w2 = new (buffer) Widget(100);
    w2->show();
    w2->~Widget();

    return 0;
}
```

**输出：**
```
构造 Widget(42)
Widget ID: 42
析构 Widget(42)
构造 Widget(100)
Widget ID: 100
析构 Widget(100)
```

## 二、具体用法

### 2.1 在预分配堆内存上使用

```cpp
#include <iostream>
#include <new>

int main() {
    // 先分配原始内存
    void* mem = ::operator new(sizeof(std::string));

    // 在该内存上构造string
    std::string* s = new (mem) std::string("Hello Placement New");
    std::cout << "字符串: " << *s << std::endl;

    // 手动析构
    s->~basic_string();

    // 释放原始内存
    ::operator delete(mem);

    return 0;
}
```

**输出：**
```
字符串: Hello Placement New
```

### 2.2 内存池中的应用

```cpp
#include <iostream>
#include <new>
#include <vector>

class ObjectPool {
    alignas(std::max_align_t) char pool[1024];
    size_t offset = 0;
public:
    void* allocate(size_t size) {
        void* ptr = pool + offset;
        offset += size;
        return ptr;
    }
};

class Item {
    int value;
public:
    Item(int v) : value(v) {
        std::cout << "Item(" << value << ") 构造" << std::endl;
    }
    ~Item() {
        std::cout << "Item(" << value << ") 析构" << std::endl;
    }
};

int main() {
    ObjectPool pool;

    Item* i1 = new (pool.allocate(sizeof(Item))) Item(1);
    Item* i2 = new (pool.allocate(sizeof(Item))) Item(2);

    i1->~Item();
    i2->~Item();

    return 0;
}
```

**输出：**
```
Item(1) 构造
Item(2) 构造
Item(1) 析构
Item(2) 析构
```

## 三、注意事项与常见陷阱

- **不能对placement new结果调用`delete`**：内存不是`new`分配的，应用`~T()`手动析构。
- **确保内存对齐**：使用`alignas`或`std::aligned_alloc`保证对齐。
- **确保内存大小足够**：`sizeof(T)`必须小于等于分配的缓冲区大小。
- **placement new不分配内存**：它只在已有内存上构造对象。
- **标准库提供了`std::launder`**（C++17）：在重用内存后获取新对象的指针。
