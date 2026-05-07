# final与优化

## 一、概念说明

`final`标记的类或虚函数允许编译器进行**devirtualization**（去虚化）优化——将虚函数调用转换为直接调用，消除vtable查表开销。

## 二、具体用法

### 2.1 final允许的优化

```cpp
#include <iostream>

class Base {
public:
    virtual int compute() const { return 1; }
    virtual ~Base() = default;
};

class Derived final : public Base {  // final类
public:
    int compute() const override { return 2; }
};

// 编译器知道Derived没有派生类
// 可以将虚调用优化为直接调用
int process(const Derived& d) {
    return d.compute();  // 编译器可以devirtualize
}

int main() {
    Derived d;
    std::cout << process(d) << std::endl;
    return 0;
}
```

**输出：**
```
2
```

### 2.2 final虚函数

```cpp
class Widget {
public:
    virtual void render() final {  // 不能被重写
        std::cout << "渲染Widget" << std::endl;
    }
};

// 编译器知道render()在所有派生类中行为相同
// 调用render()可以被devirtualize
```

## 三、注意事项与常见陷阱

- `final`类的所有虚函数本质上都是final的
- `final`虚函数可以被devirtualize
- 过度使用`final`限制可扩展性
- 编译器优化可能不需要`final`也能devirtualize（通过LTO等）
- `final`应基于设计决策而非性能使用
