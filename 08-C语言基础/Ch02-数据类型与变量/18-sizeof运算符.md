# 18 - sizeof 运算符

## 一、sizeof 基本概念

`sizeof` 是 C 语言的**编译时运算符**，用于获取类型或变量在当前平台上占用的**字节数**。

```c
printf("int 占用 %zu 字节\n", sizeof(int));        // 通常为 4
printf("double 占用 %zu 字节\n", sizeof(double));  // 通常为 8
```

---

## 二、sizeof 的两种用法

### 2.1 对类型使用（必须加括号）

```c
size_t s1 = sizeof(int);
size_t s2 = sizeof(double);
size_t s3 = sizeof(char *);
```

### 2.2 对变量或表达式使用（括号可选）

```c
int x = 42;
size_t s1 = sizeof x;        // 正确：括号可省略
size_t s2 = sizeof(x);       // 正确：加括号
size_t s3 = sizeof(int);     // 必须加括号

// 对表达式使用
size_t s4 = sizeof(x + 3.14); // sizeof(double) = 8
```

> **注意**：`sizeof` 是编译时运算，`sizeof` 表达式中的子表达式**不会被执行**。

```c
int x = 0;
sizeof(x++);    // x++ 不会执行！x 仍然是 0
printf("%d\n", x);  // 0
```

---

## 三、基本类型的 sizeof

```c
#include <stdio.h>

int main(void) {
    printf("char:        %zu 字节\n", sizeof(char));        // 1（永远）
    printf("short:       %zu 字节\n", sizeof(short));       // 通常 2
    printf("int:         %zu 字节\n", sizeof(int));         // 通常 4
    printf("long:        %zu 字节\n", sizeof(long));        // 4 或 8
    printf("long long:   %zu 字节\n", sizeof(long long));   // 通常 8
    printf("float:       %zu 字节\n", sizeof(float));       // 通常 4
    printf("double:      %zu 字节\n", sizeof(double));      // 通常 8
    printf("long double: %zu 字节\n", sizeof(long double)); // 8/12/16
    printf("_Bool:       %zu 字节\n", sizeof(_Bool));       // 1

    return 0;
}
```

> `sizeof(char)` **始终为 1**，这是 C 标准的保证。

---

## 四、指针的 sizeof

```c
// 所有指针大小相同（在同一平台上）
printf("char*:   %zu\n", sizeof(char *));    // 32位: 4, 64位: 8
printf("int*:    %zu\n", sizeof(int *));     // 32位: 4, 64位: 8
printf("void*:   %zu\n", sizeof(void *));    // 32位: 4, 64位: 8
printf("double*: %zu\n", sizeof(double *));  // 32位: 4, 64位: 8

// 函数指针大小与数据指针可能不同（但通常相同）
printf("函数指针: %zu\n", sizeof(void (*)(void)));
```

---

## 五、数组的 sizeof

### 5.1 静态数组

```c
int arr[10];
printf("数组总大小: %zu 字节\n", sizeof(arr));           // 40
printf("元素个数: %zu\n", sizeof(arr) / sizeof(arr[0])); // 10
printf("单个元素: %zu 字节\n", sizeof(arr[0]));          // 4
```

### 5.2 数组退化为指针

```c
void func(int arr[10]) {
    // arr 已退化为 int*，不是数组
    printf("sizeof(arr) = %zu\n", sizeof(arr));  // 8（指针大小）
}

int main(void) {
    int arr[10];
    printf("sizeof(arr) = %zu\n", sizeof(arr));  // 40（数组大小）
    func(arr);
    return 0;
}
```

### 5.3 多维数组

```c
int matrix[3][4];
printf("总大小: %zu\n", sizeof(matrix));           // 48 (3*4*4)
printf("行数: %zu\n", sizeof(matrix) / sizeof(matrix[0]));         // 3
printf("列数: %zu\n", sizeof(matrix[0]) / sizeof(matrix[0][0]));   // 4
```

---

## 六、结构体的 sizeof

### 6.1 基本结构体

```c
struct Simple {
    char a;      // 1 字节
    int b;       // 4 字节
    char c;      // 1 字节
};

printf("sizeof(struct Simple) = %zu\n", sizeof(struct Simple));
// 可能是 12 而不是 6！（由于内存对齐）
```

### 6.2 内存对齐的影响

```c
struct Aligned {
    char a;      // 偏移 0，占 1 字节
                 // 填充 3 字节（对齐到 4 字节边界）
    int b;       // 偏移 4，占 4 字节
    char c;      // 偏移 8，占 1 字节
                 // 填充 3 字节（结构体大小对齐到最大成员的倍数）
};
// sizeof = 12

struct Reordered {
    int b;       // 偏移 0，占 4 字节
    char a;      // 偏移 4，占 1 字节
    char c;      // 偏移 5，占 1 字节
                 // 填充 2 字节
};
// sizeof = 8（成员重排后更紧凑）
```

### 6.3 使用 #pragma pack

```c
#pragma pack(push, 1)   // 1 字节对齐
struct Packed {
    char a;
    int b;
    char c;
};
#pragma pack(pop)

printf("sizeof(struct Packed) = %zu\n", sizeof(struct Packed));  // 6
```

---

## 七、联合体的 sizeof

联合体的大小等于**最大成员的大小**（考虑对齐）：

```c
union Data {
    int i;       // 4 字节
    double d;    // 8 字节
    char c;      // 1 字节
};

printf("sizeof(union Data) = %zu\n", sizeof(union Data));  // 8
```

---

## 八、枚举的 sizeof

```c
enum Color { RED, GREEN, BLUE };
printf("sizeof(enum Color) = %zu\n", sizeof(enum Color));  // 通常 4
```

枚举的大小通常是 `int` 的大小，但 C 标准允许编译器选择更小的类型。

---

## 九、sizeof 与字符串

```c
char str1[] = "Hello";
char *str2 = "Hello";

printf("sizeof(str1) = %zu\n", sizeof(str1));  // 6（含 '\0'）
printf("sizeof(str2) = %zu\n", sizeof(str2));  // 8（指针大小）

// strlen 只计算字符数，不含 '\0'
printf("strlen(str1) = %zu\n", strlen(str1));  // 5
```

---

## 十、可变长度数组（VLA）

```c
void func(int n) {
    int vla[n];              // VLA
    printf("sizeof(vla) = %zu\n", sizeof(vla));  // n * sizeof(int)
    // sizeof 对 VLA 是运行时求值（不是编译时常量）
}
```

> 对于非 VLA，`sizeof` 是编译时常量；对于 VLA，是运行时表达式。

---

## 十一、使用 sizeof 的常见模式

```c
// 1. 计算数组元素个数
#define ARRAY_SIZE(arr) (sizeof(arr) / sizeof((arr)[0]))

int arr[] = {1, 2, 3, 4, 5};
int count = ARRAY_SIZE(arr);  // 5

// 2. 动态分配内存
int *buf = malloc(1024 * sizeof(*buf));  // 推荐：使用 *buf 而非 int

// 3. 清零结构体
struct Data data;
memset(&data, 0, sizeof(data));

// 4. 安全的字符串复制
char dest[64];
strncpy(dest, source, sizeof(dest) - 1);
dest[sizeof(dest) - 1] = '\0';
```

---

## 十二、要点总结

1. `sizeof` 返回 `size_t` 类型，使用 `%zu` 格式化输出
2. `sizeof(char)` 始终为 1
3. 所有指针的 `sizeof` 相同（32 位为 4，64 位为 8）
4. `sizeof` 不会执行表达式中的副作用
5. 数组作为函数参数会退化为指针，`sizeof` 返回指针大小
6. 结构体大小受**内存对齐**影响，可能比成员大小之和更大
7. 联合体大小等于最大成员的大小
8. 使用 `sizeof(arr) / sizeof(arr[0])` 计算数组元素个数
