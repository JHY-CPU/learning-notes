# void 指针

## 一、什么是 void 指针

`void*`（void 指针）是一种**通用指针类型**，可以指向任意类型的数据对象。它被称为"无类型指针"或"泛型指针"。

```c
int n = 42;
double d = 3.14;
char c = 'A';

void *vp;

vp = &n;   // void* 指向 int
vp = &d;   // void* 指向 double
vp = &c;   // void* 指向 char
```

## 二、void 指针的特点

### 1. 可以存储任意类型的地址

```c
int a = 10;
float f = 2.5f;
int arr[5] = {1, 2, 3, 4, 5};

void *vp;
vp = &a;    // OK
vp = &f;    // OK
vp = arr;   // OK
vp = NULL;  // OK
```

### 2. 不能直接解引用

```c
void *vp = &a;
// *vp;  // 编译错误！void* 不能解引用
```

原因：编译器不知道从地址处读取多少字节。

### 3. 不能进行指针运算

```c
void *vp = arr;
// vp++;  // 编译错误！void 的大小未定义
// vp + 1; // 编译错误！
```

原因：`void` 类型没有大小，编译器不知道移动多少字节。

> 注意：GCC 扩展允许 void* 算术运算（将 void 大小视为 1），但这不是标准 C，不应依赖此行为。

### 4. 与任何指针类型可以隐式转换

C 语言中，`void*` 与其他指针类型之间的转换不需要显式强制转换：

```c
int n = 42;
void *vp = &n;          // int* -> void*，自动转换
int *ip = vp;           // void* -> int*，自动转换
printf("%d\n", *ip);    // 42
```

> **注意**：C++ 中 `void*` 到其他类型的转换需要显式 `(int*)` 强制转换。

## 三、void 指针的典型应用

### 1. malloc/calloc/realloc 的返回值

```c
// malloc 返回 void*
int *p = (int*)malloc(sizeof(int));
// C 中可以省略 (int*)，但 C++ 中不行

// 通用做法
int *arr = malloc(n * sizeof(int));  // C 风格
int *arr2 = (int*)malloc(n * sizeof(int));  // C/C++ 兼容风格
```

### 2. 通用函数参数

```c
// 通用交换函数
void swap(void *a, void *b, size_t size) {
    void *temp = malloc(size);
    memcpy(temp, a, size);
    memcpy(a, b, size);
    memcpy(b, temp, size);
    free(temp);
}

// 使用
int x = 10, y = 20;
swap(&x, &y, sizeof(int));

double m = 1.5, n = 2.5;
swap(&m, &n, sizeof(double));
```

### 3. 通用数据结构

```c
// 通用链表节点
struct Node {
    void *data;
    struct Node *next;
};

// 创建节点
struct Node* create_node(void *data) {
    struct Node *node = malloc(sizeof(struct Node));
    node->data = data;
    node->next = NULL;
    return node;
}
```

### 4. 回调函数的上下文

```c
void for_each(int *arr, int n, void (*callback)(void*)) {
    for (int i = 0; i < n; i++) {
        callback(&arr[i]);
    }
}

void print_int(void *p) {
    printf("%d\n", *(int*)p);
}

void double_it(void *p) {
    *(int*)p *= 2;
}
```

## 四、使用 void* 的注意事项

### 1. 类型安全丢失

```c
int n = 42;
void *vp = &n;
double *dp = vp;  // 编译通过，但类型错误
printf("%f\n", *dp);  // 未定义行为！
```

### 2. 必须记住原始类型

```c
void process(void *data, char type) {
    switch (type) {
        case 'i':
            printf("%d\n", *(int*)data);
            break;
        case 'd':
            printf("%f\n", *(double*)data);
            break;
        case 's':
            printf("%s\n", (char*)data);
            break;
    }
}
```

### 3. 对齐问题

```c
char buffer[16];
void *vp = buffer;
// 将 buffer 强制转换为 double* 可能产生对齐问题
// double 通常需要 8 字节对齐
double *dp = (double*)buffer;  // 如果 buffer 未对齐，可能崩溃
```

## 五、void* 与函数指针

`void*` 和函数指针之间的转换在标准 C 中是**未定义行为**，虽然许多系统上可以工作。

```c
// 标准 C 中不保证安全
void *vp = (void*)some_function;
// typedef void (*FuncPtr)();
// FuncPtr fp = (FuncPtr)vp;  // 不保证可移植
```

## 六、关键要点总结

> **核心概念**
> - `void*` 是通用指针，可指向任意类型
> - 不能直接解引用，不能直接运算
> - 转换为具体类型后才能正常使用

> **主要用途**
> - 动态内存分配函数的返回值
> - 通用函数参数（如 qsort 的比较函数）
> - 泛型数据结构

> **注意事项**
> - 类型安全需要程序员自己保证
> - 使用时先转换为正确的具体类型
> - 注意内存对齐要求
