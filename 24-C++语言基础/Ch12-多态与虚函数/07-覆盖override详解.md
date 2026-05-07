# 覆盖（override）详解

## 一、概念说明

**覆盖**（Override）指派生类重新定义基类的虚函数，实现运行时多态。重写要求函数签名完全匹配（参数、const、引用限定符、返回类型协变）。

## 二、具体用法

### 2.1 完整签名匹配

```cpp
#include <iostream>

class Base {
public:
    virtual void func(int) const { std::cout << "Base" << std::endl; }
    virtual ~Base() = default;
};

class Derived : public Base {
public:
    // 必须签名完全匹配：参数+const
    void func(int) const override { std::cout << "Derived" << std::endl; }

    // 以下都会编译错误（有override时）：
    // void func(double) const override {}  // 参数类型不同
    // void func(int) override {}            // 缺少const
    // void func(int) const volatile override {}  // 多了volatile
};

int main() {
    Derived d;
    Base* b = &d;
    b->func(42);  // Derived
    return 0;
}
```

**输出：**
```
Derived
```

## 三、注意事项与常见陷阱

- **始终使用`override`**关键字标记重写的虚函数
- `const`和引用限定符（`&`、`&&`）必须匹配
- 返回类型可以是协变的（派生类指针/引用）
- 没有`override`时签名不匹配只会隐藏而非报错
- 函数遮蔽（hiding）和覆盖（overriding）不同
