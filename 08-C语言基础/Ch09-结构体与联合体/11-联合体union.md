# 11 - 联合体 union

## 一、联合体的基本概念

**联合体（Union）** 是一种特殊的数据类型，它的所有成员**共享同一块内存空间**。联合体的大小等于其**最大成员的大小**。

```c
union Data {
    int    i;
    float  f;
    char   str[20];
};

// sizeof(union Data) == 20（最大成员 char[20] 的大小）
```

## 二、联合体 vs 结构体

```
struct（结构体）的内存布局：       union（联合体）的内存布局：
+--------+                       +--------+
| int    |  4B                   |        |
+--------+                       | 共享   |
| float  |  4B                   | 内存   |  最大成员大小
+--------+                       | 空间   |  (20B)
| char[20]| 20B                  |        |
+--------+                       +--------+
= 28B                            = 20B
```

| 特性 | 结构体 struct | 联合体 union |
|------|-------------|-------------|
| 内存 | 各成员独立分配 | 所有成员共享 |
| 大小 | 各成员大小之和（含对齐） | 最大成员的大小 |
| 同时使用 | 所有成员可同时有效 | 只有一个成员有效 |
| 用途 | 描述复合实体 | 节省内存 / 类型双关 |

## 三、联合体的定义与使用

### 3.1 定义方式

```c
// 方式 1：单独定义
union Value {
    int   i;
    float f;
    char  c;
};

// 方式 2：定义时声明变量
union Value {
    int   i;
    float f;
} v1, v2;

// 方式 3：使用 typedef
typedef union {
    int   i;
    float f;
    char  c;
} Value;
```

### 3.2 初始化与访问

```c
#include <stdio.h>

union Data {
    int   i;
    float f;
    char  str[20];
};

int main(void) {
    union Data d;

    // 初始化时只有第一个成员有效
    d.i = 42;
    printf("d.i = %d\n", d.i);

    // 写入新成员会覆盖旧成员
    d.f = 3.14f;
    printf("d.f = %.2f\n", d.f);
    // 此时 d.i 的值已无意义（被覆盖）

    // 指定初始化器
    union Data d2 = {.str = "Hello"};
    printf("d2.str = %s\n", d2.str);

    return 0;
}
```

### 3.3 联合体的大小计算

```c
union A {
    char  a;  // 1 字节
    short b;  // 2 字节
    int   c;  // 4 字节
};
// sizeof(union A) == 4

union B {
    char  a;    // 1 字节
    double b;   // 8 字节
};
// sizeof(union B) == 8

union C {
    char a;     // 1 字节
    int  b;     // 4 字节
    char arr[5]; // 5 字节
};
// sizeof(union C) == 8（含对齐填充，最大成员为 char[5]，对齐到 4 的倍数）
```

## 四、联合体的经典使用场景

### 4.1 节省内存——变体类型

当多个数据不会同时使用时，联合体可以节省大量内存。

```c
// 不同类型的传感器，同一时刻只有一种传感器活跃
typedef struct {
    int sensor_type;  // 0=温度, 1=压力, 2=湿度
    union {
        float temperature;
        float pressure;
        float humidity;
    } reading;
} SensorData;

// 使用
SensorData s;
s.sensor_type = 0;
s.reading.temperature = 25.5;

// 节省了 sizeof(float) * 2 = 8 字节（如果不使用 union 需要 3 个 float）
```

### 4.2 类型双关（Type Punning）

通过联合体以不同方式解释同一块内存。

```c
union FloatBits {
    float f;
    unsigned int bits;
};

// 查看 float 的二进制表示
union FloatBits fb;
fb.f = 3.14f;
printf("3.14f 的二进制: 0x%08X\n", fb.bits);
```

### 4.3 网络协议解析

```c
// IP 地址：可以按字节访问，也可以按整数访问
union IPv4Addr {
    unsigned int  addr;       // 整体访问
    unsigned char octets[4];  // 按字节访问
};

union IPv4Addr ip;
ip.addr = 0xC0A80001;  // 192.168.0.1
printf("%d.%d.%d.%d\n",
       ip.octets[0], ip.octets[1],
       ip.octets[2], ip.octets[3]);
```

## 五、联合体与枚举配合

使用枚举标记联合体中哪个成员是有效的，这是安全使用联合体的最佳实践。

```c
typedef enum {
    TYPE_INT,
    TYPE_FLOAT,
    TYPE_STRING
} ValueType;

typedef struct {
    ValueType type;
    union {
        int   i;
        float f;
        char  str[64];
    } value;
} Variant;

// 安全打印
void print_variant(const Variant *v) {
    switch (v->type) {
        case TYPE_INT:
            printf("整数: %d\n", v->value.i);
            break;
        case TYPE_FLOAT:
            printf("浮点: %.2f\n", v->value.f);
            break;
        case TYPE_STRING:
            printf("字符串: %s\n", v->value.str);
            break;
    }
}

// 使用
Variant v1 = {TYPE_INT, {.i = 42}};
Variant v2 = {TYPE_FLOAT, {.f = 3.14}};
Variant v3 = {TYPE_STRING, {.str = "Hello"}};

print_variant(&v1);
print_variant(&v2);
print_variant(&v3);
```

## 六、联合体与结构体嵌套

```c
// 常见模式：结构体包含联合体
struct Packet {
    unsigned char type;
    union {
        struct {
            int x;
            int y;
        } mouse_move;
        struct {
            unsigned char key;
            unsigned char state;
        } key_event;
        struct {
            unsigned char button;
            int x, y;
        } mouse_click;
    } data;
};
```

## 七、关键要点

> **重要提示：**
>
> 1. 联合体所有成员共享内存，同一时刻**只有一个成员有效**。
> 2. 联合体大小等于最大成员的大小（含对齐填充）。
> 3. 使用枚举标记联合体中有效成员是**最佳实践**。
> 4. 联合体不能定义构造函数或析构函数（C 语言特性）。
> 5. 结构体包含联合体的模式在实际开发中非常常见。
> 6. 通过联合体进行类型双关在 C99 中是允许的（与 C++ 不同）。
