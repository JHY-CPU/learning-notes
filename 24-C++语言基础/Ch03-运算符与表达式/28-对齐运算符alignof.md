# 对齐运算符 alignof

## 一、概念说明

`alignof`（C++11）返回类型的**对齐要求**（字节数），`alignas`指定变量或类型的对齐方式。内存对齐影响结构体大小和CPU访问效率。

## 二、具体用法

### 2.1 alignof基本用法

```cpp
#include <iostream>
using namespace std;

int main() {
    // 基本类型的对齐要求
    cout << "alignof(char):   " << alignof(char) << endl;    // 1
    cout << "alignof(int):    " << alignof(int) << endl;     // 4
    cout << "alignof(double): " << alignof(double) << endl;  // 8
    cout << "alignof(void*):  " << alignof(void*) << endl;   // 8（64位）

    // 结构体的对齐等于最大成员的对齐
    struct S { char c; int i; double d; };
    cout << "alignof(S): " << alignof(S) << endl;  // 8
    cout << "sizeof(S):  " << sizeof(S) << endl;   // 16（含填充）

    // alignas指定对齐
    alignas(16) int alignedVar;
    cout << "alignas(16) int: " << alignof(decltype(alignedVar)) << endl;  // 16

    return 0;
}
```

输出：
```
alignof(char):   1
alignof(int):    4
alignof(double): 8
alignof(void*):  8
alignof(S): 8
sizeof(S):  16
alignas(16) int: 16
```

### 2.2 alignas应用

```cpp
#include <iostream>
using namespace std;

// 缓存行对齐（通常64字节）
struct alignas(64) CacheLine {
    int data[16];
};

// SIMD对齐（16字节对齐）
struct alignas(16) SIMDVector {
    float x, y, z, w;
};

int main() {
    CacheLine line;
    cout << "CacheLine对齐: " << alignof(CacheLine) << endl;
    cout << "CacheLine大小: " << sizeof(CacheLine) << endl;

    SIMDVector vec{1.0f, 2.0f, 3.0f, 4.0f};
    cout << "SIMDVector对齐: " << alignof(SIMDVector) << endl;
    cout << "SIMDVector大小: " << sizeof(SIMDVector) << endl;

    // alignas可以用于变量
    alignas(32) int arr[8];
    cout << "arr对齐: " << alignof(decltype(arr)) << endl;

    return 0;
}
```

输出：
```
CacheLine对齐: 64
CacheLine大小: 64
SIMDVector对齐: 16
SIMDVector大小: 16
arr对齐: 32
```

## 三、注意事项与常见陷阱

1. **对齐必须是2的幂**：`alignas(3)`是编译错误
2. **不能减少对齐**：`alignas(1) double`不会减少double的对齐
3. **过度对齐**：对齐超过`__STDCPP_DEFAULT_NEW_ALIGNMENT__`需要特殊分配器
4. **结构体对齐**：等于最大成员的对齐，但可以用alignas覆盖
5. **SIMD和缓存优化**：正确的对齐可以显著提升性能
