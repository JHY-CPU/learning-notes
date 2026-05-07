# 指针与constexpr

## 一、概念说明

`constexpr`上下文中的指针有特殊限制。`constexpr`指针在编译期必须指向编译期已知的地址。在`constexpr`函数中，`this`指针可被解引用，但动态分配的内存不可在编译期使用。

## 二、具体用法

### 2.1 constexpr指针

```cpp
static int global = 42;

// constexpr指针必须指向编译期已知地址
constexpr int* p = &global;  // OK：全局变量地址在编译期已知
std::cout << *p << std::endl;  // 输出: 42

// constexpr指针可修改指向的值
*p = 100;
std::cout << global << std::endl;  // 输出: 100
```

### 2.2 constexpr函数中的指针

```cpp
constexpr int sumArray(const int* arr, int n) {
    int total = 0;
    for (int i = 0; i < n; ++i) {
        total += arr[i];
    }
    return total;
}

constexpr int data[] = {1, 2, 3, 4, 5};
constexpr int result = sumArray(data, 5);
static_assert(result == 15, "should be 15");
```

### 2.3 constexpr函数中不能动态分配

```cpp
// C++20之前：constexpr函数不能使用new
// constexpr int* bad() { return new int(42); }  // C++20前错误

// C++20：允许constexpr中使用new，但结果不能用于编译期上下文
constexpr int* alloc() { return new int(42); }  // C++20 OK

// delete也可在constexpr中使用（C++20）
```

### 2.4 consteval指针

```cpp
// consteval函数中的指针操作必须全在编译期完成
consteval int deref(const int* p) {
    return *p;
}

static int x = 42;
constexpr int val = deref(&x);  // OK
std::cout << val << std::endl;  // 输出: 42
```

### 2.5 编译期数组访问

```cpp
constexpr int getElement(const int* arr, int index) {
    return arr[index];
}

constexpr int data[] = {10, 20, 30, 40, 50};
static_assert(getElement(data, 2) == 30, "data[2] = 30");
```

## 三、注意事项与常见陷阱

- `constexpr`指针指向的地址在编译期必须已知
- 局部变量地址不能用于`constexpr`指针
- C++20放宽了constexpr的限制（允许new/delete、union、try-catch等）
- `constexpr`指针本身可以是const也可以不是（取决于声明）
- 编译期指针操作受限于constexpr的其他约束
