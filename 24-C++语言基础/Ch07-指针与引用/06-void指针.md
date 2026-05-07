# void指针

## 一、概念说明

`void*`是**通用指针类型**，可以指向任意类型的数据。它不携带类型信息，使用时必须**显式转换**回具体类型。`void*`常用于泛型C API和底层内存操作。

在C++中应尽量避免使用`void*`，优先使用模板或`std::any`/`std::variant`。

## 二、具体用法

### 2.1 基本用法

```cpp
int x = 42;
double d = 3.14;

void* vp;

vp = &x;
std::cout << *static_cast<int*>(vp) << std::endl;  // 输出: 42

vp = &d;
std::cout << *static_cast<double*>(vp) << std::endl;  // 输出: 3.14
```

### 2.2 泛型函数参数

```cpp
// C风格的泛型交换函数
void swap(void* a, void* b, size_t size) {
    char* ca = static_cast<char*>(a);
    char* cb = static_cast<char*>(b);
    for (size_t i = 0; i < size; ++i) {
        std::swap(ca[i], cb[i]);
    }
}

int x = 10, y = 20;
swap(&x, &y, sizeof(int));
std::cout << x << " " << y << std::endl;  // 输出: 20 10
```

### 2.3 malloc/free

```cpp
#include <cstdlib>

// void*是malloc的返回类型
void* raw = std::malloc(sizeof(int) * 10);
int* arr = static_cast<int*>(raw);

for (int i = 0; i < 10; ++i) arr[i] = i;
std::cout << arr[5] << std::endl;  // 输出: 5

std::free(raw);
```

### 2.4 与nullptr比较

```cpp
void* vp = nullptr;

int x = 42;
vp = &x;

if (vp != nullptr) {
    std::cout << "非空指针\n";
}
// 输出: 非空指针
```

### 2.5 void指针的限制

```cpp
void* vp = &x;

// void*不能解引用
// *vp;  // 编译错误

// void*不能算术运算
// vp++;  // 编译错误（不知道类型大小）

// 必须转换为具体类型后使用
int* ip = static_cast<int*>(vp);
std::cout << *ip << std::endl;
```

## 三、注意事项与常见陷阱

- `void*`不进行类型检查，容易出错
- C++中`void*`不能隐式转换，必须用`static_cast`
- `void*`不能解引用、不能算术运算
- 现代C++中用模板替代`void*`实现泛型
- `void*`在C中用于`malloc`返回值，C++中用`new`
