# 10 - const 关键字

## 一、const 基本概念

`const` 关键字用于声明一个**不可修改**的值。它告诉编译器和程序员：这个变量/指针/参数不应该被修改。

```c
const int MAX = 100;
// MAX = 200;  // 编译错误！不能修改 const 变量
```

---

## 二、const 修饰变量

```c
const int x = 10;         // x 的值不可修改
const double pi = 3.14;   // pi 的值不可修改
const char c = 'A';       // c 的值不可修改

x = 20;    // 编译错误
pi = 3.0;  // 编译错误
```

### const 变量必须初始化

```c
const int x;       // 编译错误！const 变量必须初始化
const int x = 10;  // 正确
```

---

## 三、const 与指针

这是 `const` 最复杂也最重要的部分。理解的关键是**看 `const` 在 `*` 的哪一边**。

### 3.1 指向常量的指针（Pointer to Const）

```c
const int *p;       // 不能通过 p 修改所指向的值
int const *p;       // 同上（const 在 * 前面，两种写法等价）
```

```c
int x = 10, y = 20;
const int *p = &x;

*p = 30;         // 错误！不能通过 p 修改 x
x = 30;          // 正确！可以直接修改 x
p = &y;          // 正确！可以让 p 指向别的变量
```

### 3.2 常量指针（Const Pointer）

```c
int * const p = &x;   // p 本身不可修改（不能指向其他地址）
```

```c
int x = 10, y = 20;
int * const p = &x;

*p = 30;         // 正确！可以通过 p 修改 x
p = &y;          // 错误！不能让 p 指向别的地址
```

### 3.3 指向常量的常量指针

```c
const int * const p = &x;  // 都不可修改
```

```c
int x = 10, y = 20;
const int * const p = &x;

*p = 30;         // 错误！不能修改值
p = &y;          // 错误！不能修改指针
```

### 3.4 速记口诀

```
const 在 * 左边 → 不能修改值（指向常量的指针）
const 在 * 右边 → 不能修改指针（常量指针）

从右往左读：
const int * p    → p is a pointer to a const int
int * const p    → p is a const pointer to int
```

---

## 四、const 修饰函数参数

### 4.1 防止意外修改参数

```c
void print_array(const int *arr, int size) {
    for (int i = 0; i < size; i++) {
        printf("%d ", arr[i]);
        // arr[i] = 0;  // 编译错误！不能修改
    }
}
```

### 4.2 指针参数的 const 契约

```c
// 函数承诺不修改 str 指向的内容
size_t my_strlen(const char *str) {
    size_t len = 0;
    while (*str != '\0') {
        len++;
        str++;
    }
    return len;
}
```

### 4.3 const 指针参数

```c
// 指针本身不可变（函数内不能改变指针指向）
void process(int * const data) {
    *data = 42;    // 正确：可以修改值
    // data = NULL; // 错误：不能修改指针
}
```

---

## 五、const 修饰函数返回值

```c
// 返回指向常量数据的指针
const char *get_error_message(int code) {
    static const char *errors[] = {
        "成功",
        "文件未找到",
        "权限不足",
        "未知错误"
    };
    if (code < 0 || code > 3) code = 3;
    return errors[code];
}

// 调用者不能修改返回的字符串
const char *msg = get_error_message(1);
// msg[0] = 'x';  // 编译错误
```

---

## 六、const 与类型转换

### 6.1 添加 const（安全）

```c
int x = 10;
const int *p = &x;   // 正确：添加 const 是安全的
```

### 6.2 去除 const（危险）

```c
const int x = 10;
int *p = (int *)&x;  // 去除 const（通过强制转换）
*p = 20;             // 未定义行为！x 实际上是 const
```

> **警告**：去除 `const` 后修改原本是 `const` 的变量会导致**未定义行为**。

---

## 七、const 与数组

```c
const int arr[] = {1, 2, 3, 4, 5};
arr[0] = 10;    // 编译错误！数组元素不可修改

// 数组参数中的 const
void fill_zeros(int *arr, int n) {
    for (int i = 0; i < n; i++) arr[i] = 0;
}

void read_only(const int *arr, int n) {
    // arr[i] = 0;  // 编译错误
}
```

---

## 八、const 在多文件编程中的应用

```c
// ===== header.h =====
// 声明为 const，多个文件可以安全共享
extern const int MAX_BUFFER_SIZE;
extern const char * const VERSION;

// ===== config.c =====
const int MAX_BUFFER_SIZE = 4096;
const char * const VERSION = "1.0.0";
```

---

## 九、const 的最佳实践

### 9.1 尽可能使用 const

```c
// 好：标记不修改的参数
void print_name(const char *name);

// 好：标记不修改的局部变量
const int result = calculate(x, y);

// 好：标记不修改的返回值
const Data *get_data(void);
```

### 9.2 函数参数的 const 约定

| 参数意图 | 声明方式 |
|----------|----------|
| 不修改值 | `const int *p` 或 `const int x` |
| 不修改指针 | `int * const p` |
| 可修改 | `int *p` |
| 读写 | `int *p`（不加 const） |

---

## 十、要点总结

1. `const` 声明不可修改的值，在编译时进行检查
2. **指向常量的指针**（`const int *p`）：不能通过指针修改值
3. **常量指针**（`int *const p`）：不能修改指针本身
4. 口诀：`const` 在 `*` 左边保护值，在 `*` 右边保护指针
5. 函数参数加 `const` 是良好的编程习惯
6. 去除 `const` 后修改 const 对象是未定义行为
7. `const` 变量必须在定义时初始化
8. 使用 `const` 可以让编译器帮助检查错误，提高代码安全性
