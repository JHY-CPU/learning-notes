# 类型转换 dynamic_cast

## 一、概念说明

`dynamic_cast`执行**运行时类型检查**的安全向下转型。它利用RTTI（运行时类型信息）在运行时验证转换的合法性，失败时返回`nullptr`（指针）或抛出`std::bad_cast`异常（引用）。

## 二、具体用法

### 2.1 基本用法

```cpp
#include <iostream>
using namespace std;

class Animal {
public:
    virtual void speak() { cout << "Animal sound" << endl; }
    virtual ~Animal() = default;  // 虚析构函数
};

class Dog : public Animal {
public:
    void speak() override { cout << "Woof!" << endl; }
    void fetch() { cout << "Fetching ball!" << endl; }
};

class Cat : public Animal {
public:
    void speak() override { cout << "Meow!" << endl; }
    void purr() { cout << "Purrrr..." << endl; }
};

int main() {
    Animal* animals[] = {new Dog(), new Cat(), new Dog()};

    for (Animal* a : animals) {
        a->speak();

        // 尝试向下转型为Dog
        Dog* dog = dynamic_cast<Dog*>(a);
        if (dog != nullptr) {
            cout << "  -> 这是狗！";
            dog->fetch();
        }

        // 尝试向下转型为Cat
        Cat* cat = dynamic_cast<Cat*>(a);
        if (cat != nullptr) {
            cout << "  -> 这是猫！";
            cat->purr();
        }
    }

    // 清理
    for (Animal* a : animals) delete a;

    return 0;
}
```

输出：
```
Woof!
  -> 这是狗！Fetching ball!
Meow!
  -> 这是猫！Purrrr...
Woof!
  -> 这是狗！Fetching ball！
```

### 2.2 转换失败处理

```cpp
#include <iostream>
using namespace std;

class Base {
public:
    virtual ~Base() = default;
};

class Derived1 : public Base {
public:
    void special() { cout << "Derived1::special()" << endl; }
};

class Derived2 : public Base {
public:
    void unique() { cout << "Derived2::unique()" << endl; }
};

int main() {
    Base* bp = new Derived1();

    // 成功的转换
    Derived1* d1 = dynamic_cast<Derived1*>(bp);
    if (d1) {
        d1->special();
    }

    // 失败的转换：返回nullptr
    Derived2* d2 = dynamic_cast<Derived2*>(bp);
    if (d2 == nullptr) {
        cout << "转换失败：bp不指向Derived2" << endl;
    }

    // 引用转换失败时抛出异常
    Derived1& ref = dynamic_cast<Derived1&>(*bp);  // OK
    try {
        Derived2& ref2 = dynamic_cast<Derived2&>(*bp);
    } catch (const bad_cast& e) {
        cout << "异常: " << e.what() << endl;
    }

    delete bp;
    return 0;
}
```

输出：
```
Derived1::special()
转换失败：bp不指向Derived2
异常: std::bad_cast
```

### 2.3 交叉转型（Cross Cast）

```cpp
#include <iostream>
using namespace std;

class Printable {
public:
    virtual void print() const { cout << "Printable" << endl; }
    virtual ~Printable() = default;
};

class Serializable {
public:
    virtual void serialize() const { cout << "Serializing..." << endl; }
    virtual ~Serializable() = default;
};

class Document : public Printable, public Serializable {
public:
    void print() const override { cout << "打印文档" << endl; }
    void serialize() const override { cout << "序列化文档" << endl; }
};

int main() {
    Document doc;
    Printable* p = &doc;

    // 交叉转型：从一个基类指针转到兄弟基类
    Serializable* s = dynamic_cast<Serializable*>(p);
    if (s) {
        s->serialize();  // 成功：Document同时继承了两者
    }

    return 0;
}
```

输出：
```
序列化文档
```

### 2.4 dynamic_cast的要求

```cpp
#include <iostream>
using namespace std;

// dynamic_cast要求：
// 1. 基类必须有虚函数（至少一个）
// 2. 只能用于指针或引用
// 3. 不能用于缺乏虚函数表的类

class NoVirtual {
    int x;
};

class WithVirtual {
public:
    virtual void func() {}
    virtual ~WithVirtual() = default;
};

int main() {
    // 编译错误：NoVirtual没有虚函数
    // NoVirtual* nv = new NoVirtual();
    // NoVirtual* cast = dynamic_cast<NoVirtual*>(nv);

    // OK：有虚函数
    WithVirtual* wv = new WithVirtual();
    void* vp = dynamic_cast<void*>(wv);  // 返回对象的完整地址
    cout << "转换到void*成功" << endl;

    // dynamic_cast的性能开销
    // 需要查询虚函数表，比static_cast慢

    delete wv;
    return 0;
}
```

输出：
```
转换到void*成功
```

## 三、注意事项与常见陷阱

1. **需要虚函数**：基类必须至少有一个虚函数，否则`dynamic_cast`编译错误
2. **性能开销**：运行时类型检查有额外开销，性能敏感代码慎用
3. **RTTI必须开启**：某些项目可能关闭了RTTI（`-fno-rtti`），这时`dynamic_cast`不可用
4. **避免过度使用**：频繁使用`dynamic_cast`通常是设计问题，应考虑多态设计
5. **指针返回nullptr，引用抛异常**：转换失败的行为取决于操作对象是指针还是引用
