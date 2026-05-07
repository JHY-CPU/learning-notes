# SFINAE进阶

## 一、概念说明

SFINAE进阶技巧包括检测表达式有效性（Expression SFINAE）、使用`void_t`进行通用类型检测、以及组合多个SFINAE条件。这些技巧是实现`type_traits`的基础，也是泛型库开发的核心技能。

## 二、具体用法

### 2.1 Expression SFINAE

```cpp
// 检测类型是否有size()成员函数
template <typename T, typename = void>
struct has_size : std::false_type {};

template <typename T>
struct has_size<T, std::void_t<decltype(std::declval<T>().size())>>
    : std::true_type {};

// 检测是否有begin()迭代器
template <typename T, typename = void>
struct is_iterable : std::false_type {};

template <typename T>
struct is_iterable<T, std::void_t<
    decltype(std::declval<T>().begin()),
    decltype(std::declval<T>().end())
>> : std::true_type {};

int main() {
    std::cout << std::boolalpha;
    std::cout << has_size<std::string>::value << std::endl;  // true
    std::cout << has_size<int>::value << std::endl;          // false
    std::cout << is_iterable<std::vector<int>>::value << std::endl; // true
    std::cout << is_iterable<int>::value << std::endl;              // false
}
```

### 2.2 检测运算符有效性

```cpp
// 检测类型是否支持 operator<<
template <typename T, typename = void>
struct is_printable : std::false_type {};

template <typename T>
struct is_printable<T, std::void_t<decltype(std::declval<std::ostream&>() << std::declval<T>())>>
    : std::true_type {};

// 检测是否支持 operator+
template <typename T, typename = void>
struct has_plus : std::false_type {};

template <typename T>
struct has_plus<T, std::void_t<decltype(std::declval<T>() + std::declval<T>())>>
    : std::true_type {};

int main() {
    std::cout << is_printable<int>::value << std::endl;          // true
    std::cout << is_printable<std::vector<int>>::value << std::endl; // false
    std::cout << has_plus<int>::value << std::endl;              // true
    std::cout << has_plus<std::string>::value << std::endl;      // true
}
```

### 2.3 组合SFINAE条件

```cpp
// 同时要求多个条件
template <typename T>
constexpr bool is_container_v = has_size<T>::value && is_iterable<T>::value;

// 使用enable_if组合条件
template <typename T>
std::enable_if_t<is_container_v<T>, void>
print_container(const T& container) {
    std::cout << "[ ";
    for (const auto& item : container) {
        std::cout << item << " ";
    }
    std::cout << "]" << std::endl;
}

// 值类型的打印
template <typename T>
std::enable_if_t<!is_container_v<T> && is_printable<T>::value, void>
print_container(const T& value) {
    std::cout << value << std::endl;
}

int main() {
    std::vector<int> v{1, 2, 3};
    print_container(v);      // [ 1 2 3 ]
    print_container(42);     // 42
}
```

### 2.4 实现is_same_v

```cpp
// 自定义is_same实现
template <typename T, typename U>
struct is_same : std::false_type {};

template <typename T>
struct is_same<T, T> : std::true_type {};

// C++17 变量模板
template <typename T, typename U>
constexpr bool is_same_v = is_same<T, U>::value;

int main() {
    std::cout << std::boolalpha;
    std::cout << is_same_v<int, int> << std::endl;       // true
    std::cout << is_same_v<int, double> << std::endl;    // false
}
```

## 三、注意事项与常见陷阱

- `std::declval<T>()`只能在decltype/decltype等不求值上下文中使用
- Expression SFINAE是检测成员函数、运算符等最强大的方式
- `std::void_t`（C++17）可以接受多个参数，任一无效则SFINAE失败
- 继承`std::true_type`/`std::false_type`可以方便地获得`value`和`::type`
- 组合多个检测trait时注意短路求值不适用于模板参数
- C++20 Concepts可以替代大部分SFINAE场景，提供更好的错误信息
