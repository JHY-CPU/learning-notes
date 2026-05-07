# const引用

## 一、概念说明

`const T&`（const引用）是C++中最重要的参数传递方式之一。它同时具备：**无拷贝开销**（像引用一样高效）、**不可修改**（像值传递一样安全）、**可绑定临时对象**（延长临时对象寿命）。

## 二、具体用法

### 2.1 基本const引用

```cpp
int x = 42;
const int& ref = x;

std::cout << ref << std::endl;  // 输出: 42
// ref = 50;  // 编译错误：不能通过const引用修改
```

### 2.2 延长临时对象寿命

```cpp
// const引用可以绑定到临时对象，并延长其寿命
const std::string& s = std::string("Hello") + " World";
std::cout << s << std::endl;  // 输出: Hello World
// s在作用域结束前有效

// 非const引用不能绑定临时对象
// std::string& bad = std::string("Hello");  // 编译错误
```

### 2.3 函数参数（推荐）

```cpp
// 推荐：const引用传递大型对象
void printInfo(const std::string& name, const std::vector<int>& data) {
    std::cout << name << ": ";
    for (int x : data) std::cout << x << " ";
    std::cout << std::endl;
}

std::vector<int> v = {1, 2, 3};
printInfo("Numbers", v);
// 输出: Numbers: 1 2 3
```

### 2.4 与值传递对比

```cpp
// 值传递：拷贝整个对象（开销大）
void byValue(std::string s) { }

// const引用：无拷贝（推荐）
void byRef(const std::string& s) { }

std::string large(10000, 'X');
byValue(large);  // 拷贝10000个字符
byRef(large);    // 仅传递引用
```

### 2.5 const引用绑定非const对象

```cpp
int x = 42;
const int& ref = x;  // OK：非const → const

x = 50;  // 通过原变量修改OK
std::cout << ref << std::endl;  // 输出: 50

// ref = 60;  // 不能通过const引用修改
```

## 三、注意事项与常见陷阱

- `const T&`可以绑定到字面量：`const int& r = 42;`
- 绑定到临时对象时，对象寿命延长至引用的作用域结束
- 基本类型（int, double等）传值和传const引用性能差异很小
- `const T&`不能用于需要修改参数的场景
- 引用绑定的不是临时对象本身，而是编译器创建的临时对象
