# Concepts进阶

## 一、概念说明

Concepts进阶包括内置concept的使用、自定义复杂concept的定义、约束偏序（Constraint Subsumption）规则、以及concept在类模板和成员函数中的应用。

## 二、具体用法

### 2.1 标准库内置concept

```cpp
#include <concepts>

// 常用内置concept
template <std::integral T>
T gcd(T a, T b) {
    while (b != 0) { T t = b; b = a % b; a = t; }
    return a;
}

template <std::floating_point T>
T circle_area(T radius) {
    return static_cast<T>(3.14159265358979) * radius * radius;
}

int main() {
    std::cout << gcd(12, 8) << std::endl;           // 4
    std::cout << circle_area(2.0) << std::endl;      // 12.5664
}
```

### 2.2 自定义复杂concept

```cpp
// 复合concept：可比较且可哈希
template <typename T>
concept HashableAndComparable = requires(T a, T b) {
    { a == b } -> std::convertible_to<bool>;
    { a < b } -> std::convertible_to<bool>;
    { std::hash<T>{}(a) } -> std::convertible_to<std::size_t>;
};

// 多类型concept
template <typename T, typename U>
concept ConvertibleTo = std::is_convertible_v<T, U> &&
                        requires(T val) { static_cast<U>(val); };

template <HashableAndComparable T>
class HashedSet {
    std::unordered_set<T> data;
public:
    void insert(const T& val) { data.insert(val); }
    bool contains(const T& val) const { return data.count(val) > 0; }
    std::size_t size() const { return data.size(); }
};

int main() {
    HashedSet<int> hs;
    hs.insert(42); hs.insert(100);
    std::cout << hs.contains(42) << std::endl;   // 1 (true)
    std::cout << hs.contains(50) << std::endl;   // 0 (false)
}
```

### 2.3 约束偏序（Constraint Subsumption）

```cpp
// 更特化的concept包含更通用的
template <typename T>
concept Integral = std::is_integral_v<T>;

template <typename T>
concept SignedIntegral = Integral<T> && std::is_signed_v<T>;

// 重载决议：更特化的concept优先
template <Integral T>
void process(T val) {
    std::cout << "Integral: " << val << std::endl;
}

template <SignedIntegral T>
void process(T val) {
    std::cout << "SignedIntegral: " << val << std::endl;
}

int main() {
    process(42);     // SignedIntegral（更特化）
    process(42U);    // Integral（unsigned不满足SignedIntegral）
}
```

### 2.4 concept用于类模板

```cpp
template <std::floating_point T>
class Vector3 {
    T x, y, z;
public:
    Vector3(T x = 0, T y = 0, T z = 0) : x(x), y(y), z(z) {}

    T length() const {
        return std::sqrt(x*x + y*y + z*z);
    }

    friend std::ostream& operator<<(std::ostream& os, const Vector3& v) {
        return os << "(" << v.x << ", " << v.y << ", " << v.z << ")";
    }
};

int main() {
    Vector3<double> v(3.0, 4.0, 0.0);
    std::cout << v << " 长度=" << v.length() << std::endl;
    // (3, 4, 0) 长度=5
    // Vector3<int> vi;  // 编译错误：int不满足floating_point
}
```

## 三、注意事项与常见陷阱

- 约束偏序规则：如果concept A的约束包含concept B，则A比B更特化
- 包含关系（subsumption）基于逻辑蕴含，不是简单的参数个数
- 多个concept用`&&`组合时，所有都必须满足
- `std::same_as<T>`检查精确类型匹配，`std::convertible_to<T,U>`检查可转换性
- concept不能有模板参数的默认值
- 建议将concept放在独立的头文件中以便复用
- 标准库的`<concepts>`头文件提供了大量预定义concept
