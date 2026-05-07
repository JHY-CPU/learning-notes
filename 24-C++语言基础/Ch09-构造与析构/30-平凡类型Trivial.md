# 平凡类型（Trivial）

## 一、概念说明

**平凡类型**（Trivial Type）是可以用`memcpy`安全复制的类型，其特殊成员函数由编译器默认生成且不执行任何用户定义的操作。平凡类型在内存中可以像C结构体一样直接复制。

### 1.1 判断标准

类型`T`是平凡类型，当且仅当：
- 有平凡的默认构造函数
- 有平凡的拷贝/移动构造函数
- 有平凡的拷贝/移动赋值运算符
- 有平凡的析构函数
- 没有虚函数和虚基类

## 二、具体用法

### 2.1 判断平凡类型

```cpp
#include <iostream>
#include <type_traits>

struct Trivial { int x; double y; };           // 平凡
struct NonTrivial { int x; NonTrivial() {} };  // 非平凡（自定义构造）

int main() {
    std::cout << "Trivial is_trivial: "
              << std::is_trivial_v<Trivial> << std::endl;
    std::cout << "NonTrivial is_trivial: "
              << std::is_trivial_v<NonTrivial> << std::endl;
    std::cout << "int is_trivial: "
              << std::is_trivial_v<int> << std::endl;
    return 0;
}
```

**输出：**
```
Trivial is_trivial: 1
NonTrivial is_trivial: 0
int is_trivial: 1
```

### 2.2 POD类型（C++11/14/20变化）

```cpp
#include <iostream>
#include <type_traits>

// C++11: POD = 平凡 + 标准布局
struct POD {
    int a;
    double b;
};

struct NotPOD {
    virtual void f() {}  // 有虚函数，非POD
};

int main() {
    std::cout << "POD is_pod: " << std::is_pod_v<POD> << std::endl;
    std::cout << "NotPOD is_pod: " << std::is_pod_v<NotPOD> << std::endl;
    // C++20中is_pod被废弃，改用is_trivial && is_standard_layout
    return 0;
}
```

**输出：**
```
POD is_pod: 1
NotPOD is_pod: 0
```

## 三、注意事项与常见陷阱

- 平凡类型可以安全使用`memcpy`复制
- C++20废弃`std::is_pod`，改用`std::is_trivial_v<T> && std::is_standard_layout_v<T>`
- 自定义析构函数会破坏平凡性（即使函数体为空）
- 平凡类型没有用户定义的特殊成员函数
- 平凡类型适合作为底层数据结构的元素类型
