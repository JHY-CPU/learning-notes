# 内联函数（inline）

## 一、概念说明

`inline`关键字建议编译器将函数调用处替换为函数体本身，从而**消除函数调用开销**。注意：`inline`只是一个**建议**，编译器可以选择忽略。

在现代C++中，`inline`更重要的语义是**允许多个翻译单元中定义相同的函数**（即放宽ODR规则），这使得在头文件中定义函数成为可能。

## 二、具体用法

### 2.1 基本内联函数

```cpp
// 在头文件中定义inline函数
inline int square(int x) {
    return x * x;
}

// 编译器可能将 square(5) 替换为 5 * 5
int result = square(5);
std::cout << result << std::endl;
// 输出: 25
```

### 2.2 类内定义的成员函数自动内联

```cpp
class Circle {
    double radius;
public:
    // 类内定义的成员函数默认为inline
    double area() const { return 3.14159 * radius * radius; }
};

// 等价于显式声明
class Circle {
    double radius;
public:
    double area() const;  // 类外声明
};

inline double Circle::area() const {  // 类外定义加inline
    return 3.14159 * radius * radius;
}
```

### 2.3 constexpr隐含inline

```cpp
// constexpr函数自动具有inline属性
constexpr int factorial(int n) {
    return (n <= 1) ? 1 : n * factorial(n - 1);
}
```

## 三、注意事项与常见陷阱

- `inline`是建议而非命令，编译器会自行决定是否内联
- 递归函数、过大的函数通常不会被内联
- `inline`函数必须在每个使用它的翻译单元中定义（通常放头文件）
- 过度内联会导致**代码膨胀**，反而降低缓存命中率
- 现代编译器的优化器会自动内联小函数，手动标记往往不必要
- `inline`不等同于`static`，二者语义不同
