# std::type_traits详解

## 一、概念说明

`<type_traits>`头文件提供了一组模板类，用于在编译期查询和转换类型属性。这些trait分为三大类：**类型检查**、**类型转换**和**类型修改**。它们是SFINAE、编译期分支和泛型编程的基础工具。

## 二、具体用法

### 2.1 类型检查trait

```cpp
int main() {
    std::cout << std::boolalpha;

    // 基本类型检查
    std::cout << std::is_integral<int>::value << std::endl;         // true
    std::cout << std::is_integral<double>::value << std::endl;      // false
    std::cout << std::is_floating_point<float>::value << std::endl; // true
    std::cout << std::is_pointer<int*>::value << std::endl;         // true
    std::cout << std::is_array<int[5]>::value << std::endl;         // true
    std::cout << std::is_class<std::string>::value << std::endl;    // true
    std::cout << std::is_enum<std::errc>::value << std::endl;       // true

    // C++14 _v 简化
    std::cout << std::is_same_v<int, int32_t> << std::endl;        // true
    std::cout << std::is_same_v<int, long> << std::endl;           // false (平台相关)
    std::cout << std::is_const_v<const int> << std::endl;          // true
}
```

### 2.2 类型关系检查

```cpp
struct Base {};
struct Derived : Base {};

int main() {
    std::cout << std::boolalpha;
    std::cout << std::is_base_of_v<Base, Derived> << std::endl;    // true
    std::cout << std::is_convertible_v<int, double> << std::endl;  // true
    std::cout << std::is_convertible_v<double, int> << std::endl;  // true (有损转换)
    std::cout << std::is_convertible_v<int*, void*> << std::endl;  // true
}
```

### 2.3 类型修改trait

```cpp
int main() {
    // remove_const / add_const
    using T1 = std::remove_const_t<const int>;     // int
    using T2 = std::add_const_t<int>;              // const int

    // remove_pointer / add_pointer
    using T3 = std::remove_pointer_t<int*>;        // int
    using T4 = std::add_pointer_t<int>;            // int*

    // remove_reference / add_lvalue_reference
    using T5 = std::remove_reference_t<int&>;      // int
    using T6 = std::add_lvalue_reference_t<int>;   // int&

    // decay：去除cv引用，数组→指针，函数→函数指针
    using T7 = std::decay_t<const int&>;           // int
    using T8 = std::decay_t<int[3]>;               // int*
    using T9 = std::decay_t<void(int)>;            // void(*)(int)

    // common_type：推导公共类型
    using T10 = std::common_type_t<int, double>;   // double

    std::cout << std::is_same_v<T1, int> << std::endl;     // true
    std::cout << std::is_same_v<T8, int*> << std::endl;    // true
    std::cout << std::is_same_v<T10, double> << std::endl;  // true
}
```

### 2.4 条件类型选择

```cpp
// std::conditional：编译期三目运算符
template <bool IsBig>
using NumberType = std::conditional_t<IsBig, int64_t, int32_t>;

// 根据类型大小选择容器
template <typename T>
using SmartPtr = std::conditional_t<
    std::is_trivially_destructible_v<T>,
    std::unique_ptr<T>,    // 简单类型用unique_ptr
    std::shared_ptr<T>     // 复杂类型用shared_ptr
>;

int main() {
    NumberType<false> small_num = 42;        // int32_t
    NumberType<true> big_num = 1LL << 40;    // int64_t

    SmartPtr<int> p1 = std::make_unique<int>(42);
    SmartPtr<std::string> p2 = std::make_shared<std::string>("hello");

    std::cout << sizeof(small_num) << std::endl;  // 4
    std::cout << sizeof(big_num) << std::endl;    // 8
}
```

## 三、注意事项与常见陷阱

- `_v`后缀是C++17引入的变量模板简化（如`is_same_v<T,U>`代替`is_same<T,U>::value`）
- `decay_t`是最常用的类型清理工具，类似函数参数的类型退化
- `is_same`对const/volatile敏感，比较前可能需要`decay`
- `remove_const`只移除顶层const，不影响指针指向的const
- `common_type`的结果可能依赖参数顺序（如`common_type_t<int, double>`是double）
- `conditional_t`的两个分支类型都会被实例化，即使未被选择
