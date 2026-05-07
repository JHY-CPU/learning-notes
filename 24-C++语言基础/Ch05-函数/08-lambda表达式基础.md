# Lambda表达式基础

## 一、概念说明

Lambda表达式是C++11引入的**匿名函数**机制，允许在表达式位置直接定义函数对象。它由捕获列表、参数列表、返回类型和函数体组成，极大简化了回调和短小函数的写法。

基本语法：`[捕获列表](参数列表) -> 返回类型 { 函数体 }`

## 二、具体用法

### 2.1 基本lambda

```cpp
// 最简单的lambda
auto hello = []() { std::cout << "Hello Lambda!" << std::endl; };
hello();
// 输出: Hello Lambda!

// 带参数的lambda
auto add = [](int a, int b) { return a + b; };
std::cout << "3 + 5 = " << add(3, 5) << std::endl;
// 输出: 3 + 5 = 8
```

### 2.2 捕获列表

```cpp
int multiplier = 3;

// 值捕获：拷贝变量到lambda中
auto times = [multiplier](int x) { return x * multiplier; };
std::cout << times(10) << std::endl;
// 输出: 30

// 引用捕获：直接引用外部变量
int counter = 0;
auto increment = [&counter]() { counter++; };
increment();
increment();
std::cout << "counter = " << counter << std::endl;
// 输出: counter = 2

// 捕获所有变量
auto allByValue = [=]() { return multiplier; };    // 全部值捕获
auto allByRef = [&]() { counter++; };              // 全部引用捕获
```

### 2.3 在STL算法中使用

```cpp
std::vector<int> nums = {5, 2, 8, 1, 9};

// 排序（降序）
std::sort(nums.begin(), nums.end(),
    [](int a, int b) { return a > b; });

for (int n : nums) std::cout << n << " ";
// 输出: 9 8 5 2 1
```

## 三、注意事项与常见陷阱

- 值捕获的变量默认为`const`，修改需加`mutable`
- 引用捕获要注意被引用对象的生命周期，避免悬垂引用
- `this`指针可通过`[this]`或`[*this]`（C++17）捕获
- 空捕获列表`[]`的lambda可转换为函数指针
- 捕获的变量在lambda**创建时**确定值（值捕获）或绑定（引用捕获）
