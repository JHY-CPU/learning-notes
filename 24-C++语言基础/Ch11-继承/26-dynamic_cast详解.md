# dynamic_cast详解

## 一、概念说明

`dynamic_cast`（C++标准 §7.6.1.6）用于**安全的向下转型**（downcast）和**交叉转型**（crosscast）。它在运行时通过RTTI检查类型是否合法，失败时返回`nullptr`（指针）或抛出`std::bad_cast`（引用）。目标类型必须是多态类型（至少一个虚函数）。

### 1.1 dynamic_cast vs 其他转型

| 转型 | 检查时机 | 安全性 | 开销 | 适用场景 |
|------|---------|--------|------|---------|
| `dynamic_cast` | 运行时 | 安全 | 高（RTTI） | 向下/交叉转型 |
| `static_cast` | 编译时 | 不检查 | 无 | 向上转型/确定类型 |
| `reinterpret_cast` | 编译时 | 不检查 | 无 | 底层位模式转换 |

## 二、具体用法

### 2.1 安全向下转型（指针）

```cpp
#include <iostream>

class Base {
public:
    virtual ~Base() = default;  // 必须有虚函数
};

class Derived : public Base {
public:
    void derivedOnly() { std::cout << "Derived特有方法" << std::endl; }
};

class Other : public Base {
public:
    void otherOnly() { std::cout << "Other特有方法" << std::endl; }
};

int main() {
    Base* b1 = new Derived();
    Base* b2 = new Other();

    // 安全向下转型
    Derived* d1 = dynamic_cast<Derived*>(b1);
    if (d1) {
        d1->derivedOnly();  // OK
        std::cout << "b1是Derived" << std::endl;
    }

    Derived* d2 = dynamic_cast<Derived*>(b2);
    if (!d2) {
        std::cout << "b2不是Derived（d2=nullptr）" << std::endl;
    }

    delete b1;
    delete b2;
    return 0;
}
```

### 2.2 引用转型（失败抛异常）

```cpp
#include <iostream>
#include <stdexcept>

class Base { public: virtual ~Base() = default; };
class Derived : public Base {};

void process(Base& b) {
    try {
        Derived& d = dynamic_cast<Derived&>(b);
        std::cout << "转型成功" << std::endl;
    } catch (const std::bad_cast& e) {
        std::cout << "转型失败: " << e.what() << std::endl;
    }
}

int main() {
    Derived d;
    Other o;
    process(d);  // 转型成功
    process(o);  // 抛出std::bad_cast
    return 0;
}
```

### 2.3 交叉转型（多继承）

```cpp
#include <iostream>

class A { public: virtual ~A() = default; };
class B { public: virtual ~B() = default; };

class C : public A, public B {
public:
    void cMethod() { std::cout << "C::cMethod()" << std::endl; }
};

int main() {
    C* c = new C();
    A* a = c;  // 向上转型到A

    // 交叉转型：从A*转到B*（通过C的完整对象）
    B* b = dynamic_cast<B*>(a);
    if (b) {
        std::cout << "A* → B* 交叉转型成功" << std::endl;
    }

    delete c;
    return 0;
}
```

### 2.4 层次结构中的安全遍历

```cpp
#include <iostream>
#include <vector>
#include <memory>

class Component {
public:
    virtual ~Component() = default;
    virtual void render() const = 0;
};

class Button : public Component {
    std::string label;
public:
    Button(const std::string& l) : label(l) {}
    void render() const override {
        std::cout << "按钮: [" << label << "]" << std::endl;
    }
    void click() { std::cout << label << " 被点击" << std::endl; }
};

class Container : public Component {
public:
    std::vector<std::unique_ptr<Component>> children;
    void render() const override {
        std::cout << "容器(" << children.size() << "个子组件)" << std::endl;
        for (const auto& c : children) c->render();
    }
    void add(std::unique_ptr<Component> c) {
        children.push_back(std::move(c));
    }
};

void clickAllButtons(const Component* root) {
    // 尝试转为Button
    if (auto* btn = dynamic_cast<const Button*>(root)) {
        const_cast<Button*>(btn)->click();
        return;
    }
    // 尝试转为Container递归
    if (auto* container = dynamic_cast<const Container*>(root)) {
        for (const auto& child : container->children)
            clickAllButtons(child.get());
    }
}

int main() {
    auto root = std::make_unique<Container>();
    root->add(std::make_unique<Button>("确定"));
    root->add(std::make_unique<Button>("取消"));
    root->render();
    clickAllButtons(root.get());
    return 0;
}
```

## 三、注意事项与常见陷阱

1. **`dynamic_cast`需要虚函数**：运行时类型检查依赖RTTI/vtable
2. **转型失败：指针返回`nullptr`，引用抛出`std::bad_cast`**：引用版无法表示"空"
3. **交叉转型只在多继承中需要**：同一对象有多个基类子对象
4. **`dynamic_cast`有运行时开销**：频繁调用影响性能
5. **优先使用虚函数多态避免`dynamic_cast`**：多态比类型检查更优雅
6. **在性能关键代码中考虑`static_cast`**：确定类型时用static_cast更快
