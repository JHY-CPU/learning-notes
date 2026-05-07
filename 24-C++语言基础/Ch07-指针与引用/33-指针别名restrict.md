# 指针别名与restrict

## 一、概念说明

**指针别名**（Pointer Aliasing）是指两个或多个指针指向同一内存位置。编译器为保守处理别名问题可能放弃某些优化。`restrict`（C99关键字，C++中作为编译器扩展）向编译器保证指针没有别名，允许更激进的优化。

## 二、具体用法

### 2.1 别名问题

```cpp
void add(int* a, int* b, int* result) {
    // 编译器不能假设a,b,result指向不同内存
    // 因为a和result可能是同一指针
    *result = *a + *b;
}

// 如果调用 add(&x, &y, &x)，结果依赖执行顺序
```

### 2.2 GCC/Clang的__restrict__

```cpp
// __restrict__告诉编译器此指针无别名
void addOptimized(int* __restrict__ a,
                  int* __restrict__ b,
                  int* __restrict__ result) {
    // 编译器可优化为并行加载
    *result = *a + *b;
}

int x = 3, y = 5, z;
addOptimized(&x, &y, &z);
std::cout << z << std::endl;  // 输出: 8
```

### 2.3 性能影响示例

```cpp
// 无restrict：每次循环重新加载
void scale(int* arr, int* factor, int n) {
    for (int i = 0; i < n; ++i) {
        arr[i] *= *factor;  // factor可能指向arr[i]，不能缓存
    }
}

// 有restrict：factor可缓存到寄存器
void scaleOpt(int* __restrict__ arr, int* __restrict__ factor, int n) {
    int f = *factor;  // 编译器可放心缓存
    for (int i = 0; i < n; ++i) {
        arr[i] *= f;
    }
}
```

### 2.4 C++中的等效方案

```cpp
// C++标准无restrict，但可通过编译器扩展或优化提示
// GCC: __restrict / __restrict__
// MSVC: __restrict

// 或通过值传递消除别名
void scaleVal(int* arr, int factor, int n) {
    for (int i = 0; i < n; ++i) {
        arr[i] *= factor;  // 无别名问题
    }
}
```

## 三、注意事项与常见陷阱

- `restrict`不是C++标准关键字，是编译器扩展
- 违反restrict的保证（实际有别名）是未定义行为
- 编译器通常通过`-O2`/`-O3`已能进行部分别名分析
- `restrict`在数值计算、SIMD优化中效果最明显
- C++的`__restrict__`语法各编译器略有不同
