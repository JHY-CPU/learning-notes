# void指针

## 一、概念说明

`void*`是**通用指针类型**（C++11标准 §3.9.2），可以指向任意类型的数据。它不携带类型信息，使用时必须**显式转换**回具体类型。`void*`常用于泛型C API和底层内存操作。

C与C++的关键区别：
- C中`void*`可隐式转换为其他指针类型
- C++中`void*`必须显式转换（使用`static_cast`）

现代C++中应尽量避免使用`void*`，优先使用模板、`std::any`、`std::variant`或类型安全的抽象。

## 二、具体用法

### 2.1 基本用法

```cpp
#include <iostream>
using namespace std;

int main() {
    int x = 42;
    double d = 3.14;

    void* vp;

    vp = &x;
    cout << *static_cast<int*>(vp) << endl;    // 42

    vp = &d;
    cout << *static_cast<double*>(vp) << endl;  // 3.14

    // void*可以指向任意类型
    string s = "hello";
    vp = &s;
    cout << *static_cast<string*>(vp) << endl;  // hello

    return 0;
}
```

### 2.2 泛型函数参数

```cpp
#include <iostream>
#include <cstring>
using namespace std;

// C风格的泛型交换函数
void swapBytes(void* a, void* b, size_t size) {
    char* ca = static_cast<char*>(a);
    char* cb = static_cast<char*>(b);
    for (size_t i = 0; i < size; ++i) {
        swap(ca[i], cb[i]);
    }
}

int main() {
    int x = 10, y = 20;
    swapBytes(&x, &y, sizeof(int));
    cout << x << " " << y << endl;  // 20 10

    double d1 = 1.1, d2 = 2.2;
    swapBytes(&d1, &d2, sizeof(double));
    cout << d1 << " " << d2 << endl;  // 2.2 1.1

    return 0;
}
```

### 2.3 malloc/free（C风格内存管理）

```cpp
#include <iostream>
#include <cstdlib>
using namespace std;

int main() {
    // void*是malloc的返回类型
    void* raw = malloc(sizeof(int) * 10);
    if (!raw) {
        cerr << "分配失败" << endl;
        return 1;
    }

    int* arr = static_cast<int*>(raw);
    for (int i = 0; i < 10; ++i) arr[i] = i * i;
    cout << "arr[5] = " << arr[5] << endl;  // 25

    free(raw);  // 必须用free释放（不是delete）

    // 现代C++替代
    auto smartArr = make_unique<int[]>(10);
    for (int i = 0; i < 10; ++i) smartArr[i] = i * i;
    // 自动释放

    return 0;
}
```

### 2.4 回调函数中的用户数据

```cpp
#include <iostream>
using namespace std;

// C风格回调：通过void*传递用户数据
typedef void (*Callback)(int event, void* userData);

void registerCallback(Callback cb, void* userData) {
    // 模拟事件
    cb(1, userData);
}

// 用户数据结构
struct AppState {
    int counter;
    string name;
};

void myCallback(int event, void* userData) {
    auto* state = static_cast<AppState*>(userData);
    state->counter++;
    cout << state->name << " 事件" << event << " 计数: " << state->counter << endl;
}

int main() {
    AppState state{0, "MyApp"};
    registerCallback(myCallback, &state);
    // MyApp 事件1 计数: 1

    return 0;
}
```

### 2.5 void指针的限制

```cpp
#include <iostream>
using namespace std;

int main() {
    int x = 42;
    void* vp = &x;

    // void*不能解引用
    // *vp;  // 编译错误

    // void*不能算术运算（不知道类型大小）
    // vp++;     // 编译错误
    // vp + 1;   // 编译错误

    // 必须转换为具体类型后使用
    int* ip = static_cast<int*>(vp);
    cout << *ip << endl;  // 42

    // void*之间可以比较
    void* vp2 = &x;
    cout << (vp == vp2) << endl;  // 1

    // void*可以与nullptr比较
    void* vp3 = nullptr;
    if (!vp3) cout << "vp3为空\n";

    return 0;
}
```

### 2.6 现代替代方案

```cpp
#include <iostream>
#include <any>
#include <variant>
#include <functional>
using namespace std;

int main() {
    // std::any（C++17）：类型安全的任意类型容器
    any a = 42;
    cout << any_cast<int>(a) << endl;  // 42
    a = "hello"s;
    cout << any_cast<string>(a) << endl;  // hello

    // std::variant（C++17）：类型安全的联合体
    variant<int, double, string> v = 42;
    cout << get<int>(v) << endl;  // 42
    v = 3.14;
    cout << get<double>(v) << endl;  // 3.14

    // 模板：编译期类型安全的泛型
    auto swapGeneric = [](auto& a, auto& b) { swap(a, b); };
    int x = 1, y = 2;
    swapGeneric(x, y);
    cout << x << " " << y << endl;  // 2 1

    return 0;
}
```

## 三、注意事项与常见陷阱

1. **void*不进行类型检查**：容易出错，转换错误类型是UB
2. **C++中void*不能隐式转换**：必须用`static_cast`（比C更严格）
3. **void*不能解引用、不能算术运算**：丢失了类型信息
4. **现代C++中用模板替代void*实现泛型**：编译期类型安全
5. **void*在C中用于malloc返回值**：C++中用new/delete或智能指针
6. **free只能释放malloc分配的内存**：不能释放new分配的，反之亦然
7. **函数指针和void*不能互转**：标准不保证函数指针和数据指针大小相同
