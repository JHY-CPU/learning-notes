# 嵌套类与PIMPL

## 一、概念说明

**PIMPL**（Pointer to Implementation）模式通过将实现细节隐藏在另一个类中，通过指针间接访问，实现**编译防火墙**——头文件的修改不会导致依赖它的文件重新编译。

## 二、具体用法

### 2.1 PIMPL基本模式

```cpp
// widget.h —— 公共接口
class Widget {
public:
    Widget();
    ~Widget();
    void doSomething();
private:
    class Impl;  // 前向声明
    std::unique_ptr<Impl> pImpl;
};

// widget.cpp —— 实现
class Widget::Impl {
public:
    int data;
    std::vector<int> cache;

    void helper() {
        std::cout << "实现细节 data=" << data << std::endl;
    }
};

Widget::Widget() : pImpl(std::make_unique<Impl>()) {
    pImpl->data = 42;
}

Widget::~Widget() = default;  // 必须在Impl定义可见处析构

void Widget::doSomething() {
    pImpl->helper();
}
```

### 2.2 PIMPL的优势

```cpp
// 修改Impl不需要重新编译使用Widget的代码
class Widget::Impl {
    // 添加新成员、修改实现
    // 不影响widget.h，不触发重编译
    int newData;
    void newMethod() {}
};
```

### 2.3 需要显式析构函数

```cpp
class MyClass {
public:
    MyClass();
    ~MyClass();  // 必须在.cpp中定义（Impl在此处完整定义）

private:
    class Impl;
    std::unique_ptr<Impl> pImpl;
};

// .cpp中
MyClass::~MyClass() = default;  // 此时Impl已完整定义
// 如果不定义析构函数，unique_ptr对不完整类型调用delete会编译错误
```

### 2.4 拷贝/移动处理

```cpp
class Widget {
public:
    Widget();
    ~Widget();

    // 深拷贝实现
    Widget(const Widget&);
    Widget& operator=(const Widget&);

    // 移动
    Widget(Widget&&) noexcept;
    Widget& operator=(Widget&&) noexcept;

private:
    class Impl;
    std::unique_ptr<Impl> pImpl;
};
```

## 三、注意事项与常见陷阱

- `unique_ptr`需要析构函数处有`Impl`的完整定义
- PIMPL增加了一次间接访问的开销（指针解引用）
- PIMPL实现拷贝需自定义深拷贝（unique_ptr不可拷贝）
- PIMPL适合大型项目减少编译时间
- 类型擦除也可通过PIMPL实现
