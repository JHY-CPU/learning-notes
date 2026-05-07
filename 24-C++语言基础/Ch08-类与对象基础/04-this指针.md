# this指针

## 一、概念说明

`this`是成员函数中的隐式指针，指向**调用该成员函数的对象**。`this`的类型是`ClassName* const`（在非const成员函数中）或`const ClassName* const`（在const成员函数中）。

## 二、具体用法

### 2.1 this的基本使用

```cpp
class Point {
    int x, y;
public:
    Point(int x, int y) : x(x), y(y) {}

    // this区分同名成员和参数
    void setXY(int x, int y) {
        this->x = x;
        this->y = y;
    }

    void print() const {
        std::cout << "(" << this->x << ", " << this->y << ")" << std::endl;
    }
};

Point p(1, 2);
p.setXY(10, 20);
p.print();
// 输出: (10, 20)
```

### 2.2 链式调用

```cpp
class Builder {
    std::string name;
    int age;
public:
    Builder& setName(const std::string& n) {
        name = n;
        return *this;  // 返回自身引用
    }
    Builder& setAge(int a) {
        age = a;
        return *this;
    }
    void show() const {
        std::cout << name << ", " << age << std::endl;
    }
};

Builder().setName("Alice").setAge(25).show();
// 输出: Alice, 25
```

### 2.3 this在const成员函数中

```cpp
class Data {
    int value;
public:
    // const成员函数中this是 const Data* const
    int getValue() const {
        // this->value = 10;  // 编译错误：不能修改
        return value;
    }

    // 非const成员函数中this是 Data* const
    void setValue(int v) {
        this->value = v;  // 可修改
    }
};
```

### 2.4 this作为参数传递

```cpp
class Node;

class Observer {
public:
    void observe(Node* node) {
        std::cout << "观察节点\n";
    }
};

class Node {
    Observer obs;
public:
    void registerSelf() {
        obs.observe(this);  // 将自身指针传递给观察者
    }
};
```

### 2.5 *this作为返回值

```cpp
class Counter {
    int count = 0;
public:
    Counter& increment() {
        ++count;
        return *this;  // 返回自身引用
    }
    int get() const { return count; }
};

Counter c;
c.increment().increment().increment();
std::cout << c.get() << std::endl;  // 输出: 3
```

## 三、注意事项与常见陷阱

- `this`不能在静态成员函数中使用（静态函数无this）
- 构造函数中`this`指向正在构造的对象
- `this`是右值，不能取`&this`的地址
- const成员函数中不能通过this修改非mutable成员
- 链式调用返回`*this`引用而非值
