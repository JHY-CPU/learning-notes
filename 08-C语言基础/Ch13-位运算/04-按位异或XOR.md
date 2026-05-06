# 04 - 按位异或（XOR）

## 概念

按位异或（Bitwise XOR）运算符 `^` 对两个操作数的每一位执行异或操作：**当两个对应位不同时结果为 1，相同时结果为 0**。

### 真值表

```
0 ^ 0 = 0
0 ^ 1 = 1
1 ^ 0 = 1
1 ^ 1 = 0
```

### 运算示例

```
  1100 1010   (202)
^ 1011 0101   (181)
----------
  0111 1111   (127)
```

## 异或的重要性质

异或运算具有以下独特性质，使其在编程中非常有用：

```
1. 自反性：a ^ a = 0       （任何数与自身异或等于0）
2. 恒等性：a ^ 0 = a       （任何数与0异或等于自身）
3. 交换律：a ^ b = b ^ a
4. 结合律：(a ^ b) ^ c = a ^ (b ^ c)
5. 对称性：如果 a ^ b = c，那么 a ^ c = b 且 b ^ c = a
```

## 经典应用

### 1. 不使用临时变量交换两个数

这是异或最经典的面试题应用。

```c
#include <stdio.h>

int main(void) {
    int a = 42, b = 99;
    printf("交换前: a = %d, b = %d\n", a, b);

    // 使用异或交换（不需要临时变量）
    a = a ^ b;
    b = a ^ b;  // b = (a^b) ^ b = a
    a = a ^ b;  // a = (a^b) ^ a = b

    printf("交换后: a = %d, b = %d\n", a, b);

    return 0;
}
```

> **注意**：此方法要求 a 和 b 不能是同一个变量（即不能是同一内存地址），否则会将值清零。

### 2. 找出数组中唯一出现一次的数字

利用"相同数异或为 0"的性质。

```c
#include <stdio.h>

int main(void) {
    // 数组中除了一个数出现一次，其余都出现两次
    int arr[] = {2, 3, 5, 3, 2};
    int n = sizeof(arr) / sizeof(arr[0]);

    int result = 0;
    for (int i = 0; i < n; i++) {
        result ^= arr[i];
    }

    printf("唯一出现一次的数: %d\n", result);  // 5
    // 解释：2^3^5^3^2 = (2^2)^(3^3)^5 = 0^0^5 = 5

    return 0;
}
```

### 3. 简单加密/解密

异或的自反性使其成为最简单的对称加密方法。

```c
#include <stdio.h>
#include <string.h>

void xor_encrypt_decrypt(char *data, const char *key, int data_len, int key_len) {
    for (int i = 0; i < data_len; i++) {
        data[i] ^= key[i % key_len];  // 循环使用密钥
    }
}

int main(void) {
    char message[] = "Hello, World!";
    char key[] = "SECRET";

    printf("原始文本: %s\n", message);

    // 加密
    xor_encrypt_decrypt(message, key, strlen(message), strlen(key));
    printf("加密后: ");
    for (int i = 0; i < (int)strlen(message); i++)
        printf("%02X ", (unsigned char)message[i]);
    printf("\n");

    // 解密（再次异或同一个密钥）
    xor_encrypt_decrypt(message, key, strlen(message), strlen(key));
    printf("解密后: %s\n", message);

    return 0;
}
```

### 4. 翻转指定位

异或可以翻转（toggle）特定位，而保持其他位不变。

```c
#include <stdio.h>

void print_binary(unsigned int n) {
    for (int i = 7; i >= 0; i--)
        printf("%d", (n >> i) & 1);
}

int main(void) {
    unsigned int value = 0b10100101;

    printf("原始值: "); print_binary(value); printf("\n");

    // 翻转第3位
    value ^= (1 << 3);
    printf("翻转第3位: "); print_binary(value); printf("\n");  // 10101101

    // 翻转第7位
    value ^= (1 << 7);
    printf("翻转第7位: "); print_binary(value); printf("\n");  // 00101101

    // 翻转低4位
    value ^= 0x0F;
    printf("翻转低4位: "); print_binary(value); printf("\n");  // 00100010

    return 0;
}
```

### 5. 判断两个数是否符号不同

```c
#include <stdio.h>

int main(void) {
    int a = 42, b = -10;

    // 如果 (a ^ b) < 0，则符号不同
    if ((a ^ b) < 0)
        printf("%d 和 %d 符号不同\n", a, b);
    else
        printf("%d 和 %d 符号相同\n", a, b);

    return 0;
}
```

### 6. 找出数组中出现奇数次的两个数

```c
#include <stdio.h>

int main(void) {
    // 数组中有两个数出现奇数次，其余都出现偶数次
    int arr[] = {2, 3, 5, 3, 2, 7};
    int n = sizeof(arr) / sizeof(arr[0]);

    // 第一步：全部异或，得到两个目标数的异或结果
    int xor_all = 0;
    for (int i = 0; i < n; i++)
        xor_all ^= arr[i];
    // xor_all = 5 ^ 7 = 0b00000010 ^ 0b00000110 = 0b00000100

    // 第二步：找到 xor_all 中任意一个为1的位
    int rightmost_set_bit = xor_all & (-xor_all);  // 取最低位的1

    // 第三步：根据该位将数组分为两组
    int num1 = 0, num2 = 0;
    for (int i = 0; i < n; i++) {
        if (arr[i] & rightmost_set_bit)
            num1 ^= arr[i];  // 该位为1的一组
        else
            num2 ^= arr[i];  // 该位为0的一组
    }

    printf("两个出现奇数次的数: %d 和 %d\n", num1, num2);  // 5 和 7

    return 0;
}
```

### 7. 简单校验和

```c
#include <stdio.h>

unsigned char xor_checksum(const unsigned char *data, int len) {
    unsigned char checksum = 0;
    for (int i = 0; i < len; i++) {
        checksum ^= data[i];
    }
    return checksum;
}

int main(void) {
    unsigned char packet[] = {0x01, 0x02, 0x03, 0x04, 0x05};
    unsigned char check = xor_checksum(packet, 5);
    printf("XOR校验和: 0x%02X\n", check);  // 0x01

    // 验证
    unsigned char packet_with_check[] = {0x01, 0x02, 0x03, 0x04, 0x05, check};
    if (xor_checksum(packet_with_check, 6) == 0)
        printf("数据校验通过\n");
    else
        printf("数据校验失败\n");

    return 0;
}
```

## 异或与置反/恢复

```
设 key 为密钥/掩码：
  加密：encrypted  = original ^ key
  解密：original   = encrypted ^ key

因为：(original ^ key) ^ key = original ^ (key ^ key) = original ^ 0 = original
```

## 关键要点

- `^` 运算：相同为 0，不同为 1
- `a ^ a = 0` 和 `a ^ 0 = a` 是最核心的两个性质
- 异或可以不借助临时变量交换两个数
- 异或可用于简单加密（对称性）：加密和解密使用相同操作
- 异或可用于找出数组中出现奇数次的元素
- 异或可以翻转特定位：`value ^= mask`
