# SFINAE基础

## 一、概念说明

SFINAE（Substitution Failure Is Not An Error）即"替换失败不是错误"。在模板参数替换过程中，如果某个替换导致无效的类型或表达式，编译器不会报错，而是静默地将该模板从候选集中移除。这一机制是实现编译期类型检测和重载选择的基础。

## 二、具体用法

### 2.1 基本SFINAE

```cpp
// 当T没有value_type成员时，这个重载会被静默移除
template <typename T>
typename T::value_type get_value(T container) {
    std::cout << "有value_type" << std::endl;
    return *container.begin();
}

// 备选方案
int get_value(...) {
    std::cout << "无value_type" << std::endl;
    return 0;
}

int main() {
    std::vector<int> v{42};
    get_value(v);   // 有value_type → 返回42

    int x = 10;
    get_value(x);   // 无value_type → 返回0
}
```

### 2.2 std::enable_if

`std::enable_if`是最常用的SFINAE工具，通过条件控制模板是否参与重载：

```cpp
// 仅对整型类型启用
template <typename T>
typename std::enable_if<std::is_integral<T>::value, T>::type
safe_divide(T a, T b) {
    std::cout << "整型除法" << std::endl;
    return b != 0 ? a / b : 0;
}

// 仅对浮点型类型启用
template <typename T>
typename std::enable_if<std::is_floating_point<T>::value, T>::type
safe_divide(T a, T b) {
    std::cout << "浮点除法" << std::endl;
    return b != 0.0 ? a / b : 0.0;
}

int main() {
    std::cout << safe_divide(10, 3) << std::endl;    // 整型除法 → 3
    std::cout << safe_divide(10.0, 3.0) << std::endl; // 浮点除法 → 3.33333
    // safe_divide("a", "b");  // 编译错误：无匹配的重载
}
```

### 2.3 enable_if的不同位置

```cpp
// 方法1：返回类型（冗长）
template <typename T>
typename std::enable_if<std::is_integral<T>::value, std::string>::type
type_name() { return "integral"; }

// 方法2：默认模板参数（更简洁）
template <typename T,
          typename = typename std::enable_if<std::is_integral<T>::value>::type>
std::string type_name_v2() { return "integral"; }

// 方法3：C++14 别名模板简化
template <typename T>
std::enable_if_t<std::is_integral<T>::value, std::string>
type_name_v3() { return "integral"; }

int main() {
    std::cout << type_name<int>() << std::endl;    // integral
    std::cout << type_name_v2<int>() << std::endl; // integral
    std::cout << type_name_v3<int>() << std::endl; // integral
}
```

### 2.4 检测类型成员

```cpp
// 检测类型是否有iterator成员
template <typename T, typename = void>
struct has_iterator : std::false_type {};

template <typename T>
struct has_iterator<T, std::void_t<typename T::iterator>> : std::true_type {};

int main() {
    std::cout << std::boolalpha;
    std::cout << has_iterator<std::vector<int>>::value << std::endl;  // true
    std::cout << has_iterator<int>::value << std::endl;               // false
}
```

## 三、注意事项与常见陷阱

- SFINAE只在模板参数直接替换的上下文中生效，函数体内的错误不是SFINAE
- `std::enable_if_t`是C++14的简写，等价于`typename std::enable_if<...>::type`
- `std::void_t`（C++17）是实现类型检测的强大工具
- 多个enable_if条件可以用`&&`连接
- SFINAE错误在调试时很不直观，C++20 Concepts提供了更好的替代方案
- 确保至少有一个重载是无条件可用的，否则所有替换失败会导致编译错误
