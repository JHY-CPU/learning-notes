# sizeof面试题详解

## 一、基本数据类型

```c
#include <stdio.h>

void basic_types() {
    printf("=== 基本类型大小 ===\n");
    printf("char:        %zu\n", sizeof(char));         // 1
    printf("short:       %zu\n", sizeof(short));        // 2
    printf("int:         %zu\n", sizeof(int));           // 4
    printf("long:        %zu\n", sizeof(long));          // 4或8
    printf("long long:   %zu\n", sizeof(long long));    // 8
    printf("float:       %zu\n", sizeof(float));         // 4
    printf("double:      %zu\n", sizeof(double));        // 8
    printf("long double: %zu\n", sizeof(long double));  // 8/12/16
    printf("void*:       %zu\n", sizeof(void *));        // 4或8
    printf("size_t:      %zu\n", sizeof(size_t));        // 4或8
}
```

## 二、指针大小

```c
void pointer_sizes() {
    printf("\n=== 指针大小 ===\n");

    // 所有类型指针大小相同（在32位系统为4，64位为8）
    char  *pc;
    int   *pi;
    double *pd;
    void  *pv;
    int (**pf)(void);  // 函数指针

    printf("char*:       %zu\n", sizeof(pc));  // 8 (64位)
    printf("int*:        %zu\n", sizeof(pi));  // 8
    printf("double*:     %zu\n", sizeof(pd));  // 8
    printf("void*:       %zu\n", sizeof(pv));  // 8
    printf("函数指针:     %zu\n", sizeof(pf));  // 8

    // 指针的指针也是同样大小
    int **ppi;
    printf("int**:       %zu\n", sizeof(ppi));  // 8
}
```

## 三、数组大小

```c
void array_sizes() {
    printf("\n=== 数组大小 ===\n");

    int arr[10];
    printf("int[10]:      %zu\n", sizeof(arr));  // 40 (10*4)

    double darr[5];
    printf("double[5]:    %zu\n", sizeof(darr)); // 40 (5*8)

    char str[] = "hello";
    printf("char[]\"hello\": %zu\n", sizeof(str));  // 6 (含'\0')

    // 二维数组
    int matrix[3][4];
    printf("int[3][4]:    %zu\n", sizeof(matrix));       // 48 (3*4*4)
    printf("matrix[0]:    %zu\n", sizeof(matrix[0]));    // 16 (4*4)
    printf("matrix[0][0]: %zu\n", sizeof(matrix[0][0])); // 4

    // sizeof(数组名)得到整个数组大小
    // sizeof(指针)得到指针大小
    int *p = arr;
    printf("sizeof(arr)=%zu, sizeof(p)=%zu\n",
           sizeof(arr), sizeof(p));  // 40, 8
}
```

## 四、结构体大小（内存对齐）

```c
#include <stddef.h>

struct S1 { char a; int b; char c; };
// a[1] + pad[3] + b[4] + c[1] + pad[3] = 12

struct S2 { int b; char a; char c; };
// b[4] + a[1] + c[1] + pad[2] = 8

struct S3 { char a; char b; int c; };
// a[1] + b[1] + pad[2] + c[4] = 8

struct S4 { double d; char c; };
// d[8] + c[1] + pad[7] = 16

struct S5 { int a; double b; int c; };
// a[4] + pad[4] + b[8] + c[4] + pad[4] = 24

// 嵌套结构体
struct Inner { char x; int y; };  // 8
struct Outer { char a; struct Inner b; int c; };
// a[1] + pad[3] + b[8] + c[4] = 16? 
// 实际: a[1]+pad[3]+Inner(8)+c[4] = 16

void struct_sizes() {
    printf("\n=== 结构体大小 ===\n");
    printf("S1 (char,int,char):   %zu (预期12)\n", sizeof(struct S1));
    printf("S2 (int,char,char):   %zu (预期8)\n",  sizeof(struct S2));
    printf("S3 (char,char,int):   %zu (预期8)\n",  sizeof(struct S3));
    printf("S4 (double,char):     %zu (预期16)\n", sizeof(struct S4));
    printf("S5 (int,double,int):  %zu (预期24)\n", sizeof(struct S5));
    printf("Inner (char,int):     %zu (预期8)\n",  sizeof(struct Inner));
    printf("Outer:                %zu\n", sizeof(struct Outer));

    // 结构体成员偏移量
    printf("\nS1偏移: a=%zu, b=%zu, c=%zu\n",
           offsetof(struct S1, a),
           offsetof(struct S1, b),
           offsetof(struct S1, c));
}
```

## 五、#pragma pack影响

```c
#pragma pack(push, 1)
struct Packed1 { char a; int b; char c; };  // 6
#pragma pack(pop)

#pragma pack(push, 2)
struct Packed2 { char a; int b; char c; };  // 8
#pragma pack(pop)

#pragma pack(push, 4)
struct Packed4 { char a; int b; char c; };  // 12 (同默认)
#pragma pack(pop)

void pack_sizes() {
    printf("\n=== pack对齐影响 ===\n");
    printf("pack(1): %zu\n", sizeof(struct Packed1));  // 6
    printf("pack(2): %zu\n", sizeof(struct Packed2));  // 8
    printf("pack(4): %zu\n", sizeof(struct Packed4));  // 12
}
```

## 六、联合体大小

```c
union U1 { char a; int b; double c; };
// 大小 = max(1, 4, 8) = 8，对齐到8的倍数 = 8

union U2 { char a[5]; int b; };
// 大小 = max(5, 4) = 5，对齐到4的倍数 = 8

union U3 { char a; int b; };  
// 大小 = 4

void union_sizes() {
    printf("\n=== 联合体大小 ===\n");
    printf("U1 (char,int,double): %zu\n", sizeof(union U1));  // 8
    printf("U2 (char[5],int):     %zu\n", sizeof(union U2));  // 8
    printf("U3 (char,int):        %zu\n", sizeof(union U3));  // 4
}
```

## 七、位域大小

```c
struct BF1 { unsigned int a:1; unsigned int b:2; unsigned int c:3; };
// 共6位 < 1个int(32位) = 4字节

struct BF2 { char a:4; char b:4; };
// 共8位 = 1字节

struct BF3 { char a:3; char b:3; char c:3; };
// 跨字节：3+3=6 < 8，c需要新字节 = 2字节

struct BF4 { int a:4; int b:32; };  // b跨越两个int？
// 实现定义行为

void bitfield_sizes() {
    printf("\n=== 位域大小 ===\n");
    printf("BF1 (1,2,3 bits): %zu\n", sizeof(struct BF1));  // 4
    printf("BF2 (4,4 bits):   %zu\n", sizeof(struct BF2));  // 1
    printf("BF3 (3,3,3 bits): %zu\n", sizeof(struct BF3));  // 2
}
```

## 八、sizeof与表达式

```c
void sizeof_expression() {
    printf("\n=== sizeof表达式 ===\n");

    // sizeof不计算表达式的值
    int a = 5;
    int b = sizeof(a++);  // a不会自增！
    printf("a = %d, sizeof(a++) = %zu\n", a, b);  // a=5

    // sizeof可以对类型使用
    printf("sizeof(int) = %zu\n", sizeof(int));

    // sizeof对函数调用的返回值
    // printf返回int
    // sizeof(printf("hello"))  // "hello"不会被打印
    // 返回sizeof(int) = 4
}
```

## 九、经典面试题

```c
void classic_questions() {
    printf("\n=== 经典sizeof面试题 ===\n");

    // 题目1
    int *p1[10];       // 指针数组：10个int指针
    int (*p2)[10];     // 数组指针：指向含10个int的数组
    printf("int *p1[10]:  %zu\n", sizeof(p1));   // 80 (10*8)
    printf("int (*p2)[10]: %zu\n", sizeof(p2));   // 8 (指针)

    // 题目2：函数指针数组
    int (*fp_arr[3])(void);
    printf("int (*[3])():  %zu\n", sizeof(fp_arr));  // 24 (3*8)

    // 题目3：strlen vs sizeof
    char s1[] = "hello";
    char *s2 = "hello";
    printf("sizeof(s1)=%zu, sizeof(s2)=%zu\n", sizeof(s1), sizeof(s2));
    // s1: 6, s2: 8

    // 题目4：void大小
    // printf("%zu\n", sizeof(void));  // 非标准！GCC返回1，某些编译器报错

    // 题目5：空结构体
    struct Empty {};
    printf("空结构体: %zu\n", sizeof(struct Empty));  // GCC返回0? 1?

    // 题目6：柔性数组
    struct Flex { int len; char data[]; };
    printf("柔性数组结构体: %zu\n", sizeof(struct Flex));  // 4
}
```

## 十、汇总表

```text
| 类型/表达式         | 32位  | 64位  |
|--------------------|-------|-------|
| char               | 1     | 1     |
| short              | 2     | 2     |
| int                | 4     | 4     |
| long               | 4     | 8     |
| long long          | 8     | 8     |
| float              | 4     | 4     |
| double             | 8     | 8     |
| 指针 (任意类型)      | 4     | 8     |
| 函数指针            | 4     | 8     |
| size_t             | 4     | 8     |
*/

int main(void) {
    basic_types();
    pointer_sizes();
    array_sizes();
    struct_sizes();
    pack_sizes();
    union_sizes();
    bitfield_sizes();
    sizeof_expression();
    classic_questions();
    return 0;
}
```
