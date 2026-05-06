# 变长数组 VLA（Variable Length Array）

## 1. 概述

**变长数组（VLA）** 是C99标准引入的特性，允许使用**变量**作为数组长度，在**运行时**确定数组大小。

```c
// C89/C90：长度必须是常量表达式
int a[10];          // 合法
// int n = 10; int b[n];  // 非法！

// C99：长度可以是变量
int n = 10;
int b[n];           // 合法，VLA
```

> **注意**：C11将VLA设为**可选特性**，编译器可以不支持。MSVC（Visual Studio）一直不支持VLA。

## 2. VLA 的基本用法

### 2.1 定义与初始化

```c
#include <stdio.h>

int main(void) {
    int n;
    printf("请输入数组长度: ");
    scanf("%d", &n);

    int arr[n];  // 运行时确定大小

    // 初始化
    for (int i = 0; i < n; i++) {
        arr[i] = i * i;
    }

    // 输出
    for (int i = 0; i < n; i++) {
        printf("arr[%d] = %d\n", i, arr[i]);
    }

    return 0;
}
```

### 2.2 VLA 的特点

| 特性 | 说明 |
|------|------|
| 存储位置 | 栈上分配 |
| 大小确定时机 | 运行时 |
| 生命周期 | 离开作用域自动释放 |
| sizeof求值 | 运行时求值 |
| 初始化 | 不能在定义时用 `={}` 初始化 |

```c
void func(int n) {
    int arr[n];        // VLA，大小由参数n决定
    // int arr[n] = {0};  // 非法！VLA不能初始化

    // 手动清零
    for (int i = 0; i < n; i++) arr[i] = 0;

    printf("sizeof(arr) = %zu\n", sizeof(arr));
    // sizeof在运行时求值，结果 = n × sizeof(int)
}
```

## 3. sizeof 与 VLA

VLA的一个独特性质是 `sizeof` 在**运行时**求值。

```c
#include <stdio.h>

void print_size(int n) {
    int vla[n];
    // sizeof(vla) 在运行时计算，不是编译时
    printf("VLA长度: %d, 占用: %zu 字节\n", n, sizeof(vla));
}

int main(void) {
    print_size(5);   // VLA长度: 5, 占用: 20 字节
    print_size(10);  // VLA长度: 10, 占用: 40 字节
    return 0;
}
```

## 4. 二维 VLA

C99也支持多维VLA：

```c
#include <stdio.h>

void print_matrix(int rows, int cols) {
    int mat[rows][cols];  // 二维VLA

    // 初始化
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            mat[i][j] = i * cols + j + 1;

    // 输出
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%4d", mat[i][j]);
        }
        printf("\n");
    }
}

int main(void) {
    print_matrix(3, 4);
    return 0;
}
```

## 5. VLA 作为函数参数

```c
// VLA参数（C99写法）
int sum(int n, int arr[n]) {
    int total = 0;
    for (int i = 0; i < n; i++) total += arr[i];
    return total;
}

// 二维VLA参数
void set_identity(int n, int mat[n][n]) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            mat[i][j] = (i == j) ? 1 : 0;
}

int main(void) {
    int n = 5;
    int arr[n];
    for (int i = 0; i < n; i++) arr[i] = i + 1;

    printf("sum = %d\n", sum(n, arr));  // 15

    int size = 3;
    int m[size][size];
    set_identity(size, m);

    return 0;
}
```

## 6. VLA 的限制与风险

### 6.1 不能初始化

```c
int n = 5;
// int arr[n] = {1, 2, 3, 4, 5};  // 非法！VLA不允许初始化
int arr[n];
for (int i = 0; i < n; i++) arr[i] = i + 1;  // 只能手动赋值
```

### 6.2 栈溢出风险

VLA在栈上分配空间，如果大小过大，会导致**栈溢出**。

```c
int n = 10000000;  // 约40MB
int arr[n];        // 危险！很可能栈溢出
```

**安全实践：**
```c
#define MAX_SAFE_VLA 1000000  // 约4MB

if (n > MAX_SAFE_VLA) {
    // 使用malloc替代
    int *arr = malloc(n * sizeof(int));
    // ...
    free(arr);
} else {
    int arr[n];  // VLA安全
    // ...
}
```

### 6.3 编译器兼容性

| 编译器 | VLA支持 |
|--------|---------|
| GCC | 支持 |
| Clang | 支持 |
| MSVC | **不支持** |
| C11标准 | 可选特性 |

```c
// 检测VLA支持
#if __STDC_NO_VLA__ == 1
    #error "此编译器不支持VLA"
#endif
```

## 7. VLA vs malloc

| 特性 | VLA | malloc |
|------|-----|--------|
| 分配位置 | 栈 | 堆 |
| 分配速度 | 快 | 较慢 |
| 释放方式 | 自动 | 需要free |
| 大小限制 | 受栈大小限制（通常1-8MB） | 受系统内存限制 |
| 可调整大小 | 否 | 可用realloc |
| 初始化 | 不支持 | 可用calloc清零 |

---

## 重点总结

- VLA允许运行时确定数组大小，但**存储在栈上**，有栈溢出风险
- C99引入，C11改为**可选特性**，MSVC不支持
- VLA**不能在定义时初始化**，必须手动赋值
- `sizeof` 对VLA在**运行时求值**
- 大数组应使用 `malloc` 而非VLA
