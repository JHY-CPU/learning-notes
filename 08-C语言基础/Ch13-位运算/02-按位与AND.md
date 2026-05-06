# 02 - 按位与（AND）

## 概念

按位与（Bitwise AND）运算符 `&` 对两个操作数的每一位执行逻辑与操作：**当且仅当两个对应位都为 1 时，结果位才为 1**，否则为 0。

### 真值表

```
0 & 0 = 0
0 & 1 = 0
1 & 0 = 0
1 & 1 = 1
```

### 运算示例

```
  1100 1010   (202)
& 1011 0101   (181)
----------
  1000 0000   (128)
```

## 基本用法

```c
#include <stdio.h>

int main(void) {
    unsigned int a = 202;   // 11001010
    unsigned int b = 181;   // 10110101

    unsigned int result = a & b;
    printf("%u & %u = %u\n", a, b, result);  // 202 & 181 = 128

    return 0;
}
```

## 经典应用

### 1. 清零操作（将指定位置0）

按位与配合掩码可以将整数的指定位清零。

```c
#include <stdio.h>

void print_binary(unsigned int n) {
    for (int i = 7; i >= 0; i--)
        printf("%d", (n >> i) & 1);
}

int main(void) {
    unsigned int value = 0xFF;  // 11111111

    // 将高4位清零，保留低4位
    unsigned int result = value & 0x0F;
    printf("原始值: "); print_binary(value); printf("\n");
    printf("清零后: "); print_binary(result); printf("\n");  // 00001111

    // 将第3位清零（从0开始计数）
    unsigned int data = 0b10101100;
    data = data & ~(1 << 3);  // 第3位变为0
    printf("清除第3位: "); print_binary(data); printf("\n");  // 10100100

    return 0;
}
```

### 2. 取特定位（提取某些位的值）

```c
#include <stdio.h>

int main(void) {
    unsigned int color = 0xFFA500;  // 橙色 RGB

    // 提取红色分量（高8位）
    unsigned int red   = (color >> 16) & 0xFF;
    // 提取绿色分量（中间8位）
    unsigned int green = (color >> 8) & 0xFF;
    // 提取蓝色分量（低8位）
    unsigned int blue  = color & 0xFF;

    printf("RGB(%u, %u, %u)\n", red, green, blue);  // RGB(255, 165, 0)

    return 0;
}
```

### 3. 判断奇偶性

任何整数与 1 进行按位与操作，结果为 1 则为奇数，为 0 则为偶数。

```c
#include <stdio.h>

int main(void) {
    for (int i = 0; i <= 10; i++) {
        if (i & 1)
            printf("%d 是奇数\n", i);
        else
            printf("%d 是偶数\n", i);
    }
    return 0;
}
```

原理：奇数的二进制最低位一定是 1，偶数的最低位一定是 0。`i & 1` 等价于 `i % 2`，但速度更快。

### 4. 判断某一位是否为 1

```c
#include <stdio.h>

int main(void) {
    unsigned int value = 0b10110100;

    // 检查第2位是否为1（从0开始）
    if (value & (1 << 2))
        printf("第2位是1\n");
    else
        printf("第2位是0\n");

    // 检查第5位是否为1
    if (value & (1 << 5))
        printf("第5位是1\n");  // 这行会输出
    else
        printf("第5位是0\n");

    return 0;
}
```

### 5. 取模运算的优化

当除数是 2 的幂次时，`a & (n-1)` 等价于 `a % n`（其中 n 是 2 的幂）。

```c
#include <stdio.h>

int main(void) {
    int a = 123;

    printf("123 %% 8 = %d\n", a % 8);      // 3
    printf("123 & 7  = %d\n", a & 7);       // 3（等价于 123 % 8）

    printf("123 %% 16 = %d\n", a % 16);     // 11
    printf("123 & 15 = %d\n", a & 15);       // 11（等价于 123 % 16）

    // 原理：n 是 2 的幂，n-1 的二进制形式是全1
    // 8 = 1000，7 = 0111
    // 16 = 10000，15 = 01111
    // a & (n-1) 保留了 a 的低位部分，就是余数

    return 0;
}
```

### 6. 对齐操作

```c
#include <stdio.h>

int main(void) {
    // 将地址对齐到 4 字节边界
    unsigned int addr = 13;
    unsigned int aligned = addr & ~3;  // ~3 = ...11111100
    printf("地址 %u 对齐到4字节: %u\n", addr, aligned);  // 12

    // 将地址对齐到 8 字节边界
    unsigned int addr2 = 100;
    unsigned int aligned2 = addr2 & ~7;  // ~7 = ...11111000
    printf("地址 %u 对齐到8字节: %u\n", addr2, aligned2);  // 96

    return 0;
}
```

## 与其他操作的组合

### 按位与赋值（&=）

```c
unsigned int flags = 0xFF;
flags &= 0x0F;  // 等价于 flags = flags & 0x0F
// flags 现在是 0x0F
```

## 关键要点

- `&` 运算：两位都是 1 结果才为 1，常用于**提取位**和**清除位**
- `a & 1` 等价于 `a % 2`，用于判断奇偶
- `a & (n-1)` 在 n 为 2 的幂时等价于 `a % n`
- `value & mask` 可以提取 value 中 mask 为 1 的那些位
- `value & ~mask` 可以将 value 中 mask 为 1 的那些位清零
- 按位与常用于硬件寄存器操作、颜色分量提取、权限检查等场景
