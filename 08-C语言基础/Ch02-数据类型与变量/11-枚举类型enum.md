# 11 - 枚举类型 enum

## 一、枚举的基本概念

**枚举（enum）** 是 C 语言中一种用户定义的类型，它由一组**命名的整型常量**组成。使用枚举可以让代码更易读、更安全。

```c
enum Color {
    RED,       // 0
    GREEN,     // 1
    BLUE       // 2
};

int main(void) {
    enum Color favorite = BLUE;
    printf("favorite = %d\n", favorite);  // 2
    return 0;
}
```

---

## 二、枚举的定义方式

### 2.1 先定义类型

```c
enum Color {
    RED,
    GREEN,
    BLUE
};

enum Color c = RED;   // 使用时需要写 enum Color
```

### 2.2 定义类型的同时声明变量

```c
enum Color {
    RED, GREEN, BLUE
} color1, color2;     // 同时定义两个变量

color1 = RED;
color2 = GREEN;
```

### 2.3 匿名枚举（无类型名）

```c
enum {
    JAN = 1, FEB, MAR, APR, MAY, JUN,
    JUL, AUG, SEP, OCT, NOV, DEC
} month;

month = MAR;
```

---

## 三、枚举值的赋值规则

### 3.1 默认值

```c
enum Color {
    RED,       // 0
    GREEN,     // 1
    BLUE       // 2
};
```

### 3.2 自定义起始值

```c
enum Color {
    RED = 1,   // 1
    GREEN,     // 2
    BLUE       // 3
};
```

### 3.3 任意赋值

```c
enum Code {
    SUCCESS     = 0,
    ERROR_IO    = -1,
    ERROR_MEM   = -2,
    WARNING     = 100,
    INFO        = 200
};
```

### 3.4 混合使用

```c
enum Week {
    MON = 1,    // 1
    TUE,        // 2
    WED,        // 3
    THU,        // 4
    FRI,        // 5
    SAT = 6,    // 6（跳过了 5？不，FRI 就是 5）
    SUN         // 7
};

// 注意：SAT = 6 实际上与 FRI = 5 之后的默认值相同
```

---

## 四、typedef 与 enum 结合

使用 `typedef` 可以省去每次写 `enum` 的麻烦：

```c
typedef enum {
    JAN, FEB, MAR, APR, MAY, JUN,
    JUL, AUG, SEP, OCT, NOV, DEC
} Month;

// 之后直接使用 Month
Month current = MAR;
```

```c
// 也可以命名枚举类型
typedef enum Color {
    RED, GREEN, BLUE
} Color;

Color c = BLUE;
```

---

## 五、枚举在 switch 中的应用

```c
#include <stdio.h>

typedef enum {
    ADD, SUB, MUL, DIV
} Operation;

double calculate(Operation op, double a, double b) {
    switch (op) {
        case ADD: return a + b;
        case SUB: return a - b;
        case MUL: return a * b;
        case DIV:
            if (b == 0) {
                fprintf(stderr, "除零错误\n");
                return 0;
            }
            return a / b;
        default:
            fprintf(stderr, "未知操作\n");
            return 0;
    }
}

int main(void) {
    double result = calculate(MUL, 6, 7);
    printf("6 * 7 = %.1f\n", result);
    return 0;
}
```

---

## 六、枚举与 #define 的比较

| 特性 | enum | #define |
|------|------|---------|
| 类型安全 | 有类型 | 无类型（文本替换） |
| 调试 | 可见 | 不可见 |
| 作用域 | 遵循块作用域 | 从定义到文件末尾 |
| 值连续性 | 自动递增 | 需手动维护 |
| 范围 | 整型常量 | 任意文本 |

```c
// #define 方式
#define RED   0
#define GREEN 1
#define BLUE  2
// 缺点：没有类型关联，编译器无法检查

// enum 方式
enum Color { RED, GREEN, BLUE };
// 优点：类型关联，编译器可以检查
```

---

## 七、枚举的遍历

枚举值本质是整数，可以用循环遍历：

```c
typedef enum {
    MON, TUE, WED, THU, FRI, SAT, SUN
} Weekday;

const char *day_names[] = {
    "Monday", "Tuesday", "Wednesday",
    "Thursday", "Friday", "Saturday", "Sunday"
};

// 遍历工作日
for (Weekday d = MON; d <= FRI; d++) {
    printf("%s\n", day_names[d]);
}
```

---

## 八、枚举的位掩码用法

当需要组合多个枚举值时，可以使用位掩码：

```c
typedef enum {
    READ    = 1 << 0,   // 001 = 1
    WRITE   = 1 << 1,   // 010 = 2
    EXECUTE = 1 << 2    // 100 = 4
} Permission;

// 组合权限
int user_perm = READ | WRITE;    // 011 = 3

// 检查权限
if (user_perm & READ) {
    printf("有读权限\n");
}
if (user_perm & EXECUTE) {
    printf("有执行权限\n");  // 不会执行
}

// 添加权限
user_perm |= EXECUTE;

// 移除权限
user_perm &= ~WRITE;
```

---

## 九、枚举的注意事项

```c
// 1. 枚举值实际上是 int 类型
enum Color { RED, GREEN, BLUE };
printf("sizeof(enum Color) = %zu\n", sizeof(enum Color));  // 4（通常）

// 2. 可以将 int 赋给枚举变量（编译器可能警告）
enum Color c = 100;  // 100 不是合法的枚举值，但 C 允许

// 3. 枚举值可以重复（不建议）
enum Status {
    OK = 0,
    SUCCESS = 0,    // 与 OK 相同
    ERROR = 1
};
```

---

## 十、要点总结

1. 枚举定义了一组命名的整型常量，从 0 开始自动递增
2. 可以手动指定枚举值，未指定的成员按前一个值递增
3. 使用 `typedef` 可以简化枚举类型的使用
4. 枚举比 `#define` 更安全，有类型检查
5. 枚举适合用于 `switch` 语句和状态机
6. 枚举值可以用于位运算实现位掩码
7. 枚举类型本质上是 `int`，可以进行整数操作
8. 匿名枚举适合定义一组不需重复使用的常量
