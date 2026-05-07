# noexcept运算符

## 一、概念说明

`noexcept`既是**说明符**（声明函数不会抛异常）又是**运算符**（检查表达式是否可能抛异常）。它帮助编译器生成更高效的代码，并提供异常安全保证。

## 二、具体用法

### 2.1 noexcept说明符

```cpp
#include <iostream>
using namespace std;

// 不抛异常的函数
void safeFunc() noexcept {
    cout << "安全函数" << endl;
}

// 可能抛异常的函数
void riskyFunc() {
    throw runtime_error("出错了");
}

// 条件noexcept
template<typename T>
void process(T val) noexcept(noexcept(val.process())) {
    // 如果T::process()不抛异常，则process也不抛异常
    val.process();
}

int main() {
    safeFunc();

    try {
        riskyFunc();
    } catch (const exception& e) {
        cout << "捕获: " << e.what() << endl;
    }

    return 0;
}
```

输出：
```
安全函数
捕获: 出错了
```

### 2.2 noexcept运算符

```cpp
#include <iostream>
using namespace std;

void func1() noexcept {}
void func2() {}

int main() {
    // noexcept运算符：检查表达式是否声明为noexcept
    cout << "func1 noexcept: " << noexcept(func1()) << endl;  // true
    cout << "func2 noexcept: " << noexcept(func2()) << endl;  // false

    // 检查类型操作
    cout << "int移动构造noexcept: " << noexcept(int(declval<int>())) << endl;

    // 在移动构造中常用
    // 如果T的移动构造是noexcept，vector扩容时用移动而非拷贝
    class MyClass {
    public:
        MyClass(MyClass&&) noexcept { cout << "移动构造" << endl; }
        MyClass(const MyClass&) { cout << "拷贝构造" << endl; }
    };

    cout << "MyClass移动noexcept: "
         << noexcept(MyClass(declval<MyClass>())) << endl;

    return 0;
}
```

输出：
```
func1 noexcept: 1
func2 noexcept: 0
int移动构造noexcept: 1
移动构造
MyClass移动noexcept: 1
```

## 三、注意事项与常见陷阱

1. **noexcept是承诺**：声明noexcept的函数如果抛异常，会调用`std::terminate`
2. **移动构造应标记noexcept**：STL容器在扩容时优先使用noexcept移动
3. **析构函数默认noexcept**：C++11起析构函数默认是noexcept的
4. **条件noexcept**：在模板中根据操作是否noexcept来决定函数是否noexcept
5. **不要过度使用**：不是所有函数都需要noexcept，过度使用会降低灵活性
