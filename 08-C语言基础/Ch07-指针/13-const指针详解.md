# const 指针详解

## 一、概述

`const` 与指针的组合是 C 语言中容易混淆的知识点。有三种不同的组合方式，每种含义截然不同。

## 二、三种形式

### 1. `const int *p`（指向常量的指针）

**指针指向的数据不可通过此指针修改，但指针本身可以指向其他地址。**

```c
int x = 10, y = 20;
const int *p = &x;

// *p = 30;    // 错误！不能通过 p 修改值
p = &y;        // 正确！p 可以指向其他地址
printf("%d\n", *p);  // 20
```

等价写法：
```c
const int *p;
int const *p;   // 与上面完全相同
```

### 2. `int *const p`（常量指针）

**指针本身不可修改（不能指向其他地址），但可以通过指针修改数据。**

```c
int x = 10, y = 20;
int *const p = &x;  // 必须在定义时初始化

*p = 30;       // 正确！可以通过 p 修改值
printf("%d\n", x);  // 30

// p = &y;     // 错误！p 是常量，不能指向其他地址
```

### 3. `const int *const p`（指向常量的常量指针）

**指针和数据都不能修改。**

```c
int x = 10, y = 20;
const int *const p = &x;

// *p = 30;    // 错误！不能修改数据
// p = &y;     // 错误！不能修改指针
printf("%d\n", *p);  // 只能读取
```

## 三、记忆法则

### 口诀：const 在 `*` 的左边还是右边

```
const int *p;      // const 在 * 左边 → 修饰数据 → 数据不可改
int const *p;      // const 在 * 左边 → 同上
int *const p;      // const 在 * 右边 → 修饰指针 → 指针不可改
const int *const p; // 两边都有 → 都不可改
```

### 直观理解

```c
// 读法：从右往左读
const int *p;       // p 是指针，指向 int，const 修饰 int
int *const p;       // p 是 const，指向 int 的指针
const int *const p; // p 是 const，指向 const int 的指针
```

## 四、const 的安全性

### 赋值规则

```c
int x = 10;
const int *p1 = &x;     // OK：普通变量可以赋给 const 指针

const int cx = 10;
// int *p2 = &cx;       // 错误：const 变量不能赋给非 const 指针
const int *p3 = &cx;    // OK：const 赋给 const 指针
```

这个规则保证了 **const 的承诺不会被打破**。

### 强制去掉 const 的危险

```c
const int x = 10;
int *p = (int*)&x;      // 强制去掉 const
*p = 20;                 // 未定义行为！x 可能位于只读内存
printf("%d\n", x);       // 结果不确定
```

## 五、函数参数中的 const 指针

### 最佳实践

```c
// 输入参数：const 指针，保证函数不修改数据
void print_array(const int *arr, int n) {
    for (int i = 0; i < n; i++) {
        printf("%d ", arr[i]);
    }
}

// 输入参数：const 字符串
size_t my_strlen(const char *s) {
    const char *p = s;
    while (*p) p++;
    return p - s;
}

// 输出参数：普通指针，函数会修改数据
void init_array(int *arr, int n, int value) {
    for (int i = 0; i < n; i++) {
        arr[i] = value;
    }
}
```

### const 的好处

1. **编译期检查**：防止意外修改
2. **自文档化**：参数的 const 声明了意图
3. **编译器优化**：知道数据不变，可生成更优代码
4. **可接受更多参数**：const 指针可以接受 const 和非 const 数据

## 六、const 与字符串

```c
// 字符串字面量是 const char*
const char *str = "Hello";
// str[0] = 'h';  // 错误：修改字符串字面量是未定义行为

// 字符数组可以修改
char buf[] = "Hello";
buf[0] = 'h';  // OK
printf("%s\n", buf);  // "hello"

// const 指针 vs 指针常量
const char *p1 = "Hello";   // 不能通过 p1 修改字符串
char *const p2 = buf;       // p2 不能指向其他地方，但可以修改内容
```

## 七、常见混淆对比

```c
int x = 10, y = 20;

// 情况一：指向 const 的指针
const int *a = &x;
a = &y;      // OK
// *a = 30;  // 错误

// 情况二：const 指针
int *const b = &x;
// b = &y;   // 错误
*b = 30;     // OK

// 情况三：都是 const
const int *const c = &x;
// c = &y;   // 错误
// *c = 30;  // 错误
```

## 八、关键要点总结

> **三种形式**
> - `const int *p`：指向常量的指针，数据不可改，指针可改
> - `int *const p`：常量指针，指针不可改，数据可改
> - `const int *const p`：都不可改

> **记忆口诀**
> - const 在 `*` 左边修饰数据
> - const 在 `*` 右边修饰指针

> **编程规范**
> - 函数输入参数使用 `const` 指针
> - 不要强制去掉 const 限定
> - 字符串字面量使用 `const char*`
