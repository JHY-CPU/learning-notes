# any详解

## 一、概念说明

`std::any`是C++17引入的类型擦除容器，可以存储任意类型的单个值。它提供了运行时的多态存储，但需要类型检查和转换。

## 二、具体用法

```cpp
#include <any>
#include <iostream>
#include <string>

int main() {
    // 存储不同类型
    std::any a = 42;
    a = 3.14;
    a = std::string("hello");

    // 检查类型
    std::cout << "type: " << a.type().name() << std::endl;

    // 安全获取
    try {
        auto s = std::any_cast<std::string>(a);
        std::cout << s << std::endl;  // hello
    } catch (const std::bad_any_cast& e) {
        std::cerr << e.what() << std::endl;
    }

    // 指针版本（失败返回nullptr）
    auto* p = std::any_cast<int>(&a);
    if (p) std::cout << *p << std::endl;

    // 检查和重置
    std::cout << "has_value: " << a.has_value() << std::endl;
    a.reset();
    std::cout << "has_value: " << a.has_value() << std::endl;  // false
}
```

## 三、注意事项

- any有小对象优化（SSO），小值存在栈上
- 大对象可能有堆分配
- `any_cast<T>`类型必须精确匹配
- 性能开销高于variant（无编译时类型检查）
- 适用于需要存储未知类型的场景
