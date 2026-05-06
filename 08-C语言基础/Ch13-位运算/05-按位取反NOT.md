# 05 - 按位取反（NOT）

## 概念

按位取反（Bitwise NOT）运算符 `~` 是一个**单目运算符**，它将操作数的每一位取反：**0 变为 1，1 变为 0**。

### 真值表

```
~0 = 1
~1 = 0
```

### 运算示例

```
~ 0000 1010   (10，unsigned char)
----------
  1111 0101   (245，unsigned char)
```

## 基本用法

```c
#include <stdio.h>

void print_binary_8(unsigned char n) {
    for (int i = 7; i >= 0; i--)
        printf("%d", (n >> i) & 1);
}

int main(void) {
    unsigned char a = 10;  // 00001010
    unsigned char result = ~a;

    printf("~");
    print_binary_8(a);
    printf(" = ");
    print_binary_8(result);
    printf(" (%u)\n", result);  // 245

    return 0;
}
```

## 注意符号位

`~` 操作对**所有位**进行取反，包括符号位。这在有符号整数上使用时需要特别注意。

```c
#include <stdio.h>

int main(void) {
    int a = 0;      // 32位: 00000000 00000000 00000000 00000000
    int b = ~a;     // 32位: 11111111 11111111 11111111 11111111

    printf("~0 = %d\n", b);      // -1（补码表示）
    printf("~0 = %u\n", (unsigned int)b);  // 4294967295

    int c = 5;      // ...00000101
    int d = ~c;     // ...11111010
    printf("~5 = %d\n", d);      // -6

    // 重要关系：~x = -(x+1) 对于有符号整数成立
    printf("~5 = %d, -(5+1) = %d\n", ~c, -(c + 1));  // -6, -6

    return 0;
}
```

## 经典应用

### 1. 生成掩码

按位取反最常见的用途是生成掩码。

```c
#include <stdio.h>

int main(void) {
    // 生成低 n 位为 1、其余位为 0 的掩码
    unsigned int mask_low_8 = (1 << 8) - 1;   // 0x000000FF
    unsigned int mask_low_16 = (1 << 16) - 1;  // 0x0000FFFF

    printf("低8位掩码: 0x%08X\n", mask_low_8);
    printf("低16位掩码: 0x%08X\n", mask_low_16);

    // 使用 ~ 生成高位掩码
    unsigned int mask_high_24 = ~((1 << 8) - 1);  // 0xFFFFFF00
    printf("高24位掩码: 0x%08X\n", mask_high_24);

    // 常用技巧：生成低 n 位全 1 的掩码
    // (1U << n) - 1 生成低 n 位为 1 的掩码
    for (int n = 1; n <= 8; n++) {
        unsigned int mask = (1U << n) - 1;
        printf("低%d位掩码: 0x%02X\n", n, mask);
    }

    return 0;
}
```

### 2. 清除特定位

`value & ~mask` 的形式常用于清除指定位。

```c
#include <stdio.h>

void print_binary_8(unsigned char n) {
    for (int i = 7; i >= 0; i--)
        printf("%d", (n >> i) & 1);
}

int main(void) {
    unsigned char value = 0xFF;  // 11111111

    // 清除第3位（从0开始计数）
    value &= ~(1 << 3);
    printf("清除第3位: "); print_binary_8(value); printf("\n");  // 11110111

    // 清除低4位
    value = 0xFF;
    value &= ~0x0F;
    printf("清除低4位: "); print_binary_8(value); printf("\n");  // 11110000

    // 清除第0、2、4位
    value = 0xFF;
    value &= ~((1 << 0) | (1 << 2) | (1 << 4));
    printf("清除第0、2、4位: "); print_binary_8(value); printf("\n");  // 11101010

    return 0;
}
```

### 3. 地址对齐

```c
#include <stdio.h>

int main(void) {
    // 将地址对齐到 4 字节边界
    // ~3 = ...11111100，低2位清零
    unsigned int addr = 13;
    unsigned int aligned = addr & ~3;
    printf("地址 %u 对齐到4字节: %u\n", addr, aligned);  // 12

    // 对齐到 8 字节边界
    // ~7 = ...11111000，低3位清零
    addr = 100;
    aligned = addr & ~7;
    printf("地址 %u 对齐到8字节: %u\n", addr, aligned);  // 96

    // 对齐到 16 字节边界
    addr = 200;
    aligned = addr & ~15;
    printf("地址 %u 对齐到16字节: %u\n", addr, aligned);  // 192

    return 0;
}
```

### 4. 实现减一操作

在二进制补码系统中，`~x + 1 == -x`。

```c
#include <stdio.h>

int main(void) {
    int x = 42;
    int neg_x = ~x + 1;  // 等价于 -x

    printf("x = %d\n", x);
    printf("~x + 1 = %d\n", neg_x);  // -42
    printf("-x = %d\n", -x);         // -42

    // 这就是补码的定义：-x = ~x + 1
    return 0;
}
```

### 5. 配合其他位运算

```c
#include <stdio.h>

int main(void) {
    unsigned int flags = 0xFF;

    // ~ 与 & 配合：清除位
    flags &= ~(1 << 5);  // 清除第5位

    // ~ 与 | 配合：设置位（不常用，| 单独就够了）

    // ~ 与 ^ 配合：翻转所有位后翻转特定位（等于清除该位）
    flags ^= ~0;  // 翻转所有位

    return 0;
}
```

## ~ 与逻辑非 ! 的区别

| 对比 | 按位取反 `~` | 逻辑非 `!` |
|------|-------------|-----------|
| 操作对象 | 每个二进制位 | 整体布尔值 |
| 返回值 | 所有位取反的整数 | 0 或 1 |
| 示例 | `~5 = -6`（32位） | `!5 = 0` |

```c
int a = 5;
printf("~a = %d\n", ~a);   // -6（所有位取反）
printf("!a = %d\n", !a);   // 0（逻辑取反，非零变零）
```

## 关键要点

- `~` 是单目运算符，将操作数的每一位取反（0 变 1，1 变 0）
- 在有符号整数上，`~x = -(x+1)`，注意符号位也被取反
- `~` 最常用于生成掩码和清除特定位：`value & ~mask`
- `~` 与 `!` 完全不同：`~` 是位运算，`!` 是逻辑运算
- 对无符号整数使用 `~` 更安全，避免符号位带来的意外
