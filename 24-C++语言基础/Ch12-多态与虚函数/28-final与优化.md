# final与优化

## 一、概念说明

`final`（C++标准 §10.3/§9）标记的类或虚函数允许编译器进行**devirtualization**（去虚化）优化——将虚函数调用转换为直接调用，消除vtable查表开销。这使得虚函数调用可以被内联，接近直接调用的性能。

### 1.1 Devirtualization触发条件

| 条件 | 说明 |
|------|------|
| final类 | 编译器知道没有派生类 |
| final虚函数 | 该函数不能被进一步重写 |
| 局部对象 | 编译器看到实际类型 |
| LTO | 链接时全局分析 |
| 编译器推断 | 通过类型流分析 |

## 二、具体用法

### 2.1 final类的优化

```cpp
#include <iostream>

class Base {
public:
    virtual int compute() const { return 1; }
    virtual ~Base() = default;
};

class Derived final : public Base {
public:
    int compute() const override { return 2; }
};

// 编译器知道Derived没有派生类
// 可以将虚调用优化为直接调用
int processFinal(const Derived& d) {
    return d.compute();  // 编译器可以devirtualize → 直接调用Derived::compute
}

// 对比：非final版本
class Open : public Base {
public:
    int compute() const override { return 3; }
};

int processOpen(const Open& o) {
    return o.compute();  // 编译器可能不敢devirtualize（Open可能有派生类）
}

int main() {
    Derived d;
    std::cout << processFinal(d) << std::endl;  // 2

    Open o;
    std::cout << processOpen(o) << std::endl;   // 3

    return 0;
}
```

### 2.2 final虚函数

```cpp
#include <iostream>

class Widget {
public:
    virtual void render() final {  // 不能被重写
        std::cout << "渲染Widget" << std::endl;
    }
    virtual void resize(int w, int h) {  // 可被重写
        std::cout << "调整大小: " << w << "x" << h << std::endl;
    }
    virtual ~Widget() = default;
};

class Button final : public Widget {
    // void render() override {}  // 错误！render是final
    void resize(int w, int h) override {
        std::cout << "按钮调整: " << w << "x" << h << std::endl;
    }
};

// 编译器知道render()在所有派生类中行为相同
// 调用render()可以被devirtualize
void renderAll(Widget& w) {
    w.render();   // final函数 → 可被优化为直接调用
    w.resize(100, 50);  // 非final → 通常保留虚调用
}

int main() {
    Button b;
    renderAll(b);
    return 0;
}
```

### 2.3 编译器优化的实际效果

```cpp
#include <iostream>
#include <chrono>

class Base {
public:
    virtual int compute(int x) const { return x * 2; }
    virtual ~Base() = default;
};

class FinalDerived final : public Base {
public:
    int compute(int x) const override { return x * 3; }
};

class OpenDerived : public Base {
public:
    int compute(int x) const override { return x * 3; }
};

int main() {
    const int N = 100000000;

    FinalDerived fd;
    const Base* pb = &fd;

    // final类：编译器可devirtualize
    auto start = std::chrono::high_resolution_clock::now();
    volatile int r = 0;
    for (int i = 0; i < N; ++i)
        r = pb->compute(i);
    auto end = std::chrono::high_resolution_clock::now();
    auto finalTime = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // 非final：通常保留虚调用
    OpenDerived od;
    const Base* pb2 = &od;
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; ++i)
        r = pb2->compute(i);
    end = std::chrono::high_resolution_clock::now();
    auto openTime = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << "final派生类: " << finalTime.count() << "ms" << std::endl;
    std::cout << "非final派生类: " << openTime.count() << "ms" << std::endl;

    return 0;
}
```

## 三、注意事项与常见陷阱

1. **`final`类的所有虚函数本质上都是final的**：没有派生类就不能被重写
2. **`final`虚函数可以被devirtualize**：消除查表开销
3. **过度使用`final`限制可扩展性**：应在设计阶段决定
4. **编译器优化可能不需要`final`也能devirtualize**：通过LTO等手段
5. **`final`应基于设计决策而非性能使用**：设计意图是主要考量
6. **C++的`final`不同于Java的`final`**：Java的final是值不可变
