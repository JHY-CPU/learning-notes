# static_cast与继承

## 一、概念说明

`static_cast`（C++标准 §7.6.1.8）用于编译期的类型转换，在继承体系中用于**向上转型**（总是安全）和**向下转型**（不检查，程序员负责安全）。与`dynamic_cast`不同，`static_cast`没有运行时检查开销。

### 1.1 static_cast在继承中的用途

| 转型方向 | 安全性 | 说明 |
|----------|--------|------|
| 派生→基类（向上） | 总是安全 | 编译器隐式也支持 |
| 基类→派生（向下） | 不检查 | 程序员必须确认类型 |
| void*→具体类型 | 不检查 | C风格替代 |

## 二、具体用法

### 2.1 static_cast在继承中的使用

```cpp
#include <iostream>

class Base {
public:
    virtual ~Base() = default;
    virtual void show() { std::cout << "Base" << std::endl; }
};

class Derived : public Base {
public:
    void show() override { std::cout << "Derived" << std::endl; }
    void derivedMethod() { std::cout << "Derived特有" << std::endl; }
};

int main() {
    Derived d;
    Base& b = d;

    // 向上转型：static_cast总是安全（但通常不需要，编译器自动完成）
    Base* pb = static_cast<Base*>(&d);

    // 向下转型：程序员确认安全（无运行时检查）
    Derived* pd = static_cast<Derived*>(pb);
    pd->derivedMethod();  // OK：pb确实指向Derived

    // 危险情况：pb实际指向Base对象
    Base realBase;
    Base* pb2 = &realBase;
    Derived* pd2 = static_cast<Derived*>(pb2);  // 编译通过！
    pd2->derivedMethod();  // 未定义行为！

    return 0;
}
```

### 2.2 static_cast vs dynamic_cast

```cpp
#include <iostream>

class Base { public: virtual ~Base() = default; };
class Derived : public Base { public: void foo() {} };
class Other : public Base {};

void compareCasts() {
    Derived d;
    Base* pb = &d;

    // static_cast：编译时，不检查
    Derived* pd1 = static_cast<Derived*>(pb);  // OK，假设正确

    // dynamic_cast：运行时，检查
    Derived* pd2 = dynamic_cast<Derived*>(pb);  // OK，检查通过
    if (pd2) std::cout << "dynamic_cast成功" << std::endl;

    // 当类型不匹配时
    Other o;
    Base* pb2 = &o;

    // static_cast不检查：危险但编译通过
    Derived* pd3 = static_cast<Derived*>(pb2);  // 编译通过，运行时UB

    // dynamic_cast检查：安全
    Derived* pd4 = dynamic_cast<Derived*>(pb2);
    if (!pd4) std::cout << "dynamic_cast检测到类型不匹配" << std::endl;
}

int main() {
    compareCasts();
    return 0;
}
```

### 2.3 性能对比

```cpp
#include <iostream>
#include <chrono>

class Base { public: virtual ~Base() = default; };
class Derived : public Base {};

void benchmark() {
    Derived d;
    Base* pb = &d;

    const int N = 10000000;

    // static_cast：编译时转换，几乎零开销
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; ++i) {
        volatile Derived* p = static_cast<Derived*>(pb);
        (void)p;
    }
    auto mid = std::chrono::high_resolution_clock::now();

    // dynamic_cast：运行时检查，有开销
    for (int i = 0; i < N; ++i) {
        volatile Derived* p = dynamic_cast<Derived*>(pb);
        (void)p;
    }
    auto end = std::chrono::high_resolution_clock::now();

    auto staticTime = std::chrono::duration_cast<std::chrono::microseconds>(mid - start);
    auto dynamicTime = std::chrono::duration_cast<std::chrono::microseconds>(end - mid);

    std::cout << "static_cast:  " << staticTime.count() << " μs" << std::endl;
    std::cout << "dynamic_cast: " << dynamicTime.count() << " μs" << std::endl;
    // dynamic_cast通常慢10-100倍
}

int main() {
    benchmark();
    return 0;
}
```

### 2.4 实际应用：工厂模式中的类型确定

```cpp
#include <iostream>
#include <memory>

class Event {
public:
    virtual ~Event() = default;
};

class MouseEvent : public Event {
public:
    int x, y;
    MouseEvent(int x, int y) : x(x), y(y) {}
};

class KeyEvent : public Event {
public:
    int keyCode;
    KeyEvent(int k) : keyCode(k) {}
};

// 事件处理器基类
class Handler {
public:
    virtual ~Handler() = default;
    // 处理时已知具体类型，用static_cast
    virtual void handleImpl(const Event* e) = 0;

    // 公共接口
    void handle(const Event* e) {
        handleImpl(e);
    }
};

class MouseHandler : public Handler {
public:
    void handleImpl(const Event* e) override {
        // 已知e是MouseEvent，用static_cast更快
        auto* me = static_cast<const MouseEvent*>(e);
        std::cout << "鼠标: (" << me->x << "," << me->y << ")" << std::endl;
    }
};

int main() {
    MouseEvent me(100, 200);
    MouseHandler mh;
    mh.handle(&me);
    return 0;
}
```

## 三、注意事项与常见陷阱

1. **`static_cast`向下转型不检查安全性**：使用前必须通过其他方式确认类型
2. **`static_cast`比`dynamic_cast`快**：无运行时类型检查
3. **向上转型不需要任何cast**：编译器自动完成
4. **`static_cast`不能移除`const`**：用`const_cast`
5. **如果不确定类型，使用`dynamic_cast`并检查结果**：安全第一
6. **在虚函数内部已知类型时可用`static_cast`**：调用约定已确定类型
