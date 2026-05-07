# const指针详解

## 一、概念说明

`const`与指针的组合有三种形式，含义不同：**指向常量的指针**（`const int*`）、**常量指针**（`int* const`）、**指向常量的常量指针**（`const int* const`）。记忆口诀：`const`修饰其左边的内容。

## 二、具体用法

### 2.1 const int*（指向常量）

```cpp
const int* p = &x;
// 不能通过p修改指向的值
// *p = 20;  // 编译错误

// 但p本身可以重新指向
p = &y;  // OK

int x = 10, y = 20;
const int* p = &x;
std::cout << *p << std::endl;  // 输出: 10

p = &y;
std::cout << *p << std::endl;  // 输出: 20
```

### 2.2 int* const（常量指针）

```cpp
int x = 10, y = 20;
int* const p = &x;

// 可以通过p修改值
*p = 30;
std::cout << x << std::endl;  // 输出: 30

// 但p不能重新指向
// p = &y;  // 编译错误
```

### 2.3 const int* const（双重const）

```cpp
int x = 10;
const int* const p = &x;

// 不能通过p修改值
// *p = 20;  // 编译错误

// 不能重新指向
// p = &y;  // 编译错误

std::cout << *p << std::endl;  // 输出: 10 （可读取）
```

### 2.4 在函数参数中的应用

```cpp
// const int*：不修改指针指向的数据
void printArray(const int* arr, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        std::cout << arr[i] << " ";
        // arr[i] = 0;  // 编译错误
    }
}

int nums[] = {1, 2, 3, 4, 5};
printArray(nums, 5);
// 输出: 1 2 3 4 5
```

### 2.5 类型转换

```cpp
int x = 42;

// 非const → const：安全，隐式转换
int* p = &x;
const int* cp = p;  // OK

// const → 非const：危险，需要const_cast
const int* cp2 = &x;
int* p2 = const_cast<int*>(cp2);  // 不推荐
```

## 三、注意事项与常见陷阱

- 阅读声明时从右向左：`const int* p` = p是指针，指向const int
- `const int*`和`int const*`等价
- `const`指针可以指向非`const`对象
- 函数参数用`const T*`可防止意外修改，扩大函数适用范围
- `const_cast`移除`const`后修改原`const`对象是未定义行为
