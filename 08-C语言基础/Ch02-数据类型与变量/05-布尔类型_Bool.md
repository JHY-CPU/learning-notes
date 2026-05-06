# 05 - 布尔类型 _Bool

## 一、C99 之前的布尔表示

在 C99 标准之前，C 语言**没有专门的布尔类型**，通常使用整数来模拟：

```c
// 传统方式
int found = 0;     // 表示 false
found = 1;         // 表示 true

if (found) {
    printf("找到了\n");
}

// 或者自定义
typedef int BOOL;
#define TRUE  1
#define FALSE 0
```

---

## 二、C99 的 _Bool 类型

C99 标准引入了内置的布尔类型 `_Bool`：

```c
#include <stdbool.h>  // 可选，提供 bool、true、false 宏

int main(void) {
    _Bool flag = 1;    // true
    _Bool done = 0;    // false

    if (flag) {
        printf("flag 为真\n");
    }

    return 0;
}
```

### 2.1 _Bool 的特性

- `_Bool` 只能存储 **0** 或 **1** 两个值
- 将任何非零值赋给 `_Bool`，它会自动变为 **1**
- 将 **0** 赋给 `_Bool`，其值为 **0**
- `_Bool` 的大小为 **1 字节**

```c
_Bool b;

b = 0;     // b = 0
b = 42;    // b = 1（非零值自动转换为 1）
b = -3.14; // b = 1（非零值自动转换为 1）
b = 0.0;   // b = 0（零值转换为 0）
```

### 2.2 sizeof(_Bool)

```c
printf("sizeof(_Bool) = %zu\n", sizeof(_Bool));  // 1
```

---

## 三、stdbool.h 头文件

`<stdbool.h>` 为 `_Bool` 提供了更友好的别名：

```c
#include <stdbool.h>
```

该头文件定义了以下宏：

| 宏 | 展开为 |
|----|--------|
| `bool` | `_Bool` |
| `true` | `1` |
| `false` | `0` |

```c
#include <stdio.h>
#include <stdbool.h>

int main(void) {
    bool is_ready = true;
    bool has_error = false;

    if (is_ready && !has_error) {
        printf("系统就绪，无错误\n");
    }

    printf("true 的值: %d\n", true);    // 1
    printf("false 的值: %d\n", false);  // 0

    return 0;
}
```

> **建议**：在 C99 及以后的标准中，始终包含 `<stdbool.h>` 并使用 `bool`、`true`、`false`，使代码更清晰。

---

## 四、布尔运算

### 4.1 关系运算符（产生布尔值）

```c
bool result;

result = (5 > 3);      // true
result = (10 == 20);   // false
result = (x != 0);     // 取决于 x 的值
```

关系运算符：`>`, `<`, `>=`, `<=`, `==`, `!=`

### 4.2 逻辑运算符

```c
bool a = true, b = false;

bool c = a && b;    // 逻辑与：false
bool d = a || b;    // 逻辑或：true
bool e = !a;        // 逻辑非：false
```

| 运算符 | 名称 | 规则 |
|--------|------|------|
| `&&` | 逻辑与 | 两个都为真才为真 |
| `\|\|` | 逻辑或 | 有一个为真即为真 |
| `!` | 逻辑非 | 真变假，假变真 |

### 4.3 短路求值

```c
int x = 0, y = 5;

// && 短路：左边为假时，右边不执行
if (x != 0 && y / x > 2) {
    // 当 x == 0 时，y / x 不会执行，避免除零错误
}

// || 短路：左边为真时，右边不执行
if (x == 0 || func(x)) {
    // 当 x == 0 时，func(x) 不会执行
}
```

> 短路求值是 C 语言的重要特性，常用于**安全检查**和**条件执行**。

---

## 五、布尔类型与其他类型的转换

### 5.1 其他类型转 bool

```c
bool b1 = 42;        // true（非零为真）
bool b2 = 0;         // false（零为假）
bool b3 = -1;        // true
bool b4 = 0.001;     // true
bool b5 = 0.0;       // false
bool b6 = NULL;      // false（空指针为假）
```

### 5.2 bool 转其他类型

```c
int n = true;        // n = 1
double d = false;    // d = 0.0
char c = true;       // c = 1
```

### 5.3 C 中的"真"与"假"

在 C 语言中，以下值被视为**假（false/0）**：
- 整数 `0`
- 浮点数 `0.0`
- 空字符 `'\0'`
- 空指针 `NULL`

**所有其他值**都被视为**真**。

---

## 六、常见用法示例

```c
#include <stdio.h>
#include <stdbool.h>

// 函数返回布尔值
bool is_even(int n) {
    return n % 2 == 0;
}

bool is_positive(int n) {
    return n > 0;
}

bool in_range(int value, int min, int max) {
    return value >= min && value <= max;
}

int main(void) {
    // 用作标志位
    bool found = false;
    int arr[] = {3, 7, 1, 9, 5};
    int target = 9;

    for (int i = 0; i < 5; i++) {
        if (arr[i] == target) {
            found = true;
            break;
        }
    }

    if (found) {
        printf("找到目标值 %d\n", target);
    }

    // 组合条件
    int age = 25;
    bool has_license = true;
    bool can_drive = (age >= 18) && has_license;

    printf("可以开车: %s\n", can_drive ? "是" : "否");

    return 0;
}
```

---

## 七、_Bool 与 int 的区别

虽然 `_Bool` 本质上是整型，但它有一些独特的行为：

```c
_Bool b = 2;    // b 的值为 1（不是 2！）
int i = 2;      // i 的值为 2

// _Bool 参与运算时自动转换为 int
int result = b + 1;   // result = 2（1 + 1）
```

---

## 八、要点总结

1. C99 引入了 `_Bool` 类型，只能存储 0 或 1
2. 包含 `<stdbool.h>` 后可使用 `bool`、`true`、`false`
3. 任何非零值赋给 `_Bool` 都会自动变为 1
4. C 语言中 0、0.0、'\0'、NULL 为假，其余为真
5. `&&` 和 `||` 具有**短路求值**特性，是重要的编程技巧
6. `_Bool` 大小为 1 字节，比 `int` 更节省空间
7. 函数返回类型为 `bool` 可以使接口更清晰
8. 布尔标志变量命名建议使用 `is_`、`has_`、`can_` 等前缀
