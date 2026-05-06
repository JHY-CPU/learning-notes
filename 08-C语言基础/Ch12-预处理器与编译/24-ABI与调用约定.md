# 24 - ABI 与调用约定

## ABI 概述

ABI（Application Binary Interface，应用程序二进制接口）定义了编译后代码的二进制级规范。它决定了不同编译单元、不同编译器、甚至不同语言编写的代码如何相互交互。

## ABI 包含的内容

ABI 规定了以下方面：

| 规范 | 说明 |
|------|------|
| 数据类型大小 | int、long、指针等类型的字节数 |
| 数据对齐规则 | 结构体成员的对齐方式 |
| 字节序 | 大端/小端存储 |
| 调用约定 | 参数如何传递、返回值如何返回 |
| 名称修饰 | 符号在目标文件中的命名规则 |
| 系统调用号 | 操作系统接口编号 |
| 异常处理 | 异常如何传播和捕获 |
| 虚表布局 | C++虚函数表结构 |

## 调用约定（Calling Convention）

调用约定定义了函数调用时的底层细节。

### x86-64 System V ABI（Linux/macOS）

```c
// 参数传递规则：
// 整数参数：RDI, RSI, RDX, RCX, R8, R9
// 浮点参数：XMM0-XMM7
// 返回值：RAX（整数）, XMM0（浮点）
// 栈对齐：调用前必须 16 字节对齐
// 调用者保存：RAX, RCX, RDX, RSI, RDI, R8-R11
// 被调用者保存：RBX, RBP, R12-R15
```

```c
int add(int a, int b, int c) {
    return a + b + c;
}
```

```asm
; add(a=rdi, b=rsi, c=rdx)
add:
    lea eax, [rdi+rsi]
    add eax, edx
    ret
```

### x86-64 Microsoft ABI（Windows）

```c
// 参数传递规则：
// 整数参数：RCX, RDX, R8, R9
// 浮点参数：XMM0-XMM3
// 返回值：RAX（整数）, XMM0（浮点）
// 栈对齐：调用前必须 16 字节对齐
// 影子空间：为前 4 个参数保留 32 字节栈空间
// 调用者保存：RAX, RCX, RDX, R8-R11, XMM4-XMM5
// 被调用者保存：RBX, RBP, RDI, RSI, R12-R15, XMM6-XMM15
```

### x86（32位）调用约定

```c
// cdecl（C 默认）：
// 参数从右到左压栈
// 调用者清理栈
// 函数名修饰：_func

// stdcall（Windows API）：
// 参数从右到左压栈
// 被调用者清理栈
// 函数名修饰：_func@N（N 为参数字节数）

// fastcall：
// 前两个整数参数通过 ECX, EDX 传递
// 其余参数压栈
```

### 指定调用约定

```c
// Windows
void __stdcall api_function(int x);    // stdcall
void __cdecl c_function(int x);        // cdecl（默认）
void __fastcall fast_function(int x);  // fastcall

// GCC 属性
void __attribute__((stdcall)) api_function(int x);
void __attribute__((cdecl)) c_function(int x);
```

## 名称修饰（Name Mangling）

### C 语言

C 语言的名称修饰相对简单：

```c
// 源代码
int add(int a, int b);

// 符号名（Linux）:  add
// 符号名（Windows）: _add（32位）或 add（64位）
```

### C++（对比）

C++ 支持函数重载，需要更复杂的名称修饰：

```cpp
// C++ 源代码
int add(int a, int b);
double add(double a, double b);

// 符号名（GCC）:  _Z3addii, _Z3adddd
// 符号名（MSVC）: ?add@@YAHHH@Z, ?add@@YANNN@Z
```

### extern "C"

```c
// 在 C++ 中使用 C 的名称修饰
#ifdef __cplusplus
extern "C" {
#endif

int my_function(int x);  // 使用 C 链接（无 C++ 名称修饰）

#ifdef __cplusplus
}
#endif
```

## 二进制兼容

### 什么情况下二进制兼容

```c
// 以下修改保持二进制兼容：
// 1. 在结构体末尾添加新字段（但需要重新编译使用方）
// 2. 修改函数实现（不改变签名）
// 3. 添加新的函数

// 以下修改破坏二进制兼容：
// 1. 修改函数参数
// 2. 修改结构体字段顺序
// 3. 修改枚举值
// 4. 改变调用约定
// 5. 改变数据类型大小
```

### SONAME 版本控制

```bash
# 主版本号变化：不兼容
libfoo.so.1 → libfoo.so.2  # API 变化，不兼容

# 次版本号变化：兼容
libfoo.so.1.0 → libfoo.so.1.1  # 添加功能，兼容

# 编译时使用
gcc main.c -lfoo -o program
# 运行时链接 libfoo.so.1（主版本号）
```

## 字节序

```c
// 大端（Big-Endian）：高位字节在低地址
// 0x12345678 存储为: 12 34 56 78

// 小端（Little-Endian）：低位字节在低地址
// 0x12345678 存储为: 78 56 34 12

// x86/x64: 小端
// ARM: 可配置，默认小端
// 网络字节序: 大端
```

```c
// 检测字节序
int is_little_endian(void) {
    union {
        uint32_t i;
        uint8_t c[4];
    } test = { .i = 1 };
    return test.c[0] == 1;
}
```

## 跨 ABI 编译

```bash
# 32位和64位
gcc -m32 main.c -o main32    # 32位
gcc -m64 main.c -o main64    # 64位

# 不同 ABI（Windows MinGW）
x86_64-w64-mingw32-gcc main.c -o main.exe  # 交叉编译到 Windows
```

## 重要注意事项

> **关键点总结**：
> 1. ABI 定义了二进制级的接口规范，决定了代码的二进制兼容性
> 2. **调用约定**规定参数传递方式（寄存器 vs 栈）和栈清理责任
> 3. x86-64 Linux 使用 System V ABI，Windows 使用 Microsoft ABI
> 4. `extern "C"` 确保 C++ 代码使用 C 的简单名称修饰
> 5. **二进制兼容**是库版本管理的核心考量
> 6. 不同平台/编译器的 ABI 可能不同，导致二进制不兼容
