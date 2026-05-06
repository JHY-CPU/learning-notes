# 22 - 限定符 volatile

## 一、volatile 基本概念

`volatile` 关键字告诉编译器：这个变量的值可能在**编译器无法预料的情况下被改变**，因此不要对这个变量的访问进行优化。

```c
volatile int status_register;
```

**没有 `volatile` 时**：编译器可能假设变量值在两次读取之间不会改变，从而进行优化。

**有 `volatile` 时**：编译器每次都会从内存中重新读取变量的值。

---

## 二、编译器优化的问题

### 2.1 优化导致的 bug

```c
// 假设 0x40021000 是硬件状态寄存器的地址
int *status = (int *)0x40021000;

// 没有 volatile：编译器可能优化
while (*status & 0x01) {
    // 编译器可能认为 *status 不会变化
    // 将 *status 缓存到寄存器中，只读一次
    // 这将导致死循环！
}

// 正确写法
volatile int *status = (volatile int *)0x40021000;
while (*status & 0x01) {
    // 每次循环都会从地址 0x40021000 重新读取
}
```

### 2.2 编译器优化的两种情况

```c
// 情况 1：循环优化
int done = 0;
while (!done) {
    // 编译器可能将 done 缓存到寄存器
    // 如果 done 在外部被修改（如中断），循环可能永远不退出
}

// 情况 2：指令重排
int a = 1;
int b = 2;
// 编译器可能重排这两条指令的顺序
// 在多线程或硬件编程中，顺序可能很重要
```

---

## 三、volatile 的使用场景

### 3.1 硬件寄存器

嵌入式开发中，硬件寄存器的值可以被硬件随时修改：

```c
// 定义硬件寄存器指针
volatile uint32_t * const UART_STATUS  = (volatile uint32_t *)0x40011000;
volatile uint32_t * const UART_DATA    = (volatile uint32_t *)0x40011004;

// 读取 UART 数据
char uart_read_char(void) {
    // 等待数据就绪
    while (!(*UART_STATUS & 0x20)) {
        // UART_STATUS 可能被硬件随时修改
    }
    return (char)(*UART_DATA);
}
```

### 3.2 中断服务程序

```c
// 全局标志位，在中断中修改
volatile int data_ready = 0;

void ISR_UART(void) {
    // 中断服务程序
    data_ready = 1;   // 数据到达，设置标志
}

int main(void) {
    while (1) {
        if (data_ready) {
            // 处理数据
            data_ready = 0;
        }
        // 没有 volatile，编译器可能优化掉 data_ready 的读取
    }
}
```

### 3.3 多线程共享变量（基础用法）

```c
// 注意：volatile 不能替代互斥锁！
// 它只保证不优化，不保证原子性

volatile int stop_flag = 0;

void *worker_thread(void *arg) {
    while (!stop_flag) {
        // 工作循环
    }
    return NULL;
}

void signal_handler(int sig) {
    stop_flag = 1;  // 设置停止标志
}
```

> **警告**：`volatile` **不能**替代原子操作或互斥锁，它不提供任何同步保证。

### 3.4 setjmp/longjmp 中的变量

```c
#include <setjmp.h>

jmp_buf env;

void func(void) {
    volatile int x = 10;   // 必须用 volatile
    if (setjmp(env) == 0) {
        x = 20;
        longjmp(env, 1);   // 跳转回来时，x 的值是不确定的
    }
    // 如果 x 没有 volatile，编译器可能将其优化为寄存器变量
    // 导致 longjmp 后 x 的值不确定
    printf("x = %d\n", x);
}
```

---

## 四、volatile 的组合使用

### 4.1 volatile 与 const

```c
// 只读的状态寄存器（硬件可以修改，软件不能修改）
volatile const int *status = (volatile const int *)0x40021000;

int value = *status;      // 可以读取
// *status = 10;          // 错误！const 阻止软件修改
// 硬件仍然可以修改 *status 的值
```

### 4.2 volatile 与指针

```c
volatile int *p1;        // 指向 volatile int 的指针
                          // p1 本身不是 volatile

int * volatile p2 = &x;  // volatile 指针（指针本身是 volatile）
                          // 指向的 int 不是 volatile

volatile int * volatile p3;  // 都是 volatile
```

### 4.3 volatile 与 struct

```c
typedef struct {
    volatile uint32_t CR;    // 控制寄存器
    volatile uint32_t SR;    // 状态寄存器
    volatile uint32_t DR;    // 数据寄存器
} UART_TypeDef;

UART_TypeDef *uart = (UART_TypeDef *)0x40011000;
uart->CR = 0x01;   // 每次写入都实际执行
uint32_t sr = uart->SR;  // 每次读取都实际执行
```

---

## 五、volatile 不能做什么

| 需求 | volatile 能否满足 |
|------|-------------------|
| 防止编译器优化读写 | 能 |
| 保证操作的原子性 | **不能** |
| 保证内存顺序 | **不能**（需要内存屏障） |
| 替代互斥锁 | **不能** |
| 多线程同步 | **不能**（需要 atomic 或 mutex） |

```c
// 错误示例：volatile 不能保证原子性
volatile int counter = 0;

// 线程 1 和线程 2 同时执行
counter++;    // 这不是原子操作！
// 实际上是：读取 counter → 加 1 → 写回 counter
// 两个线程可能同时读取，导致结果不正确

// 正确方式：使用原子操作（C11）
#include <stdatomic.h>
atomic_int atomic_counter = 0;
atomic_fetch_add(&atomic_counter, 1);  // 原子操作
```

---

## 六、使用 volatile 的最佳实践

1. **嵌入式开发**：硬件寄存器必须使用 `volatile`
2. **中断标志**：在 ISR 中修改的全局变量应声明为 `volatile`
3. **不要滥用**：`volatile` 会阻止优化，影响性能
4. **不要替代同步机制**：多线程使用 `atomic` 或 `mutex`
5. **指针和类型都要考虑**：`volatile int *` 与 `int * volatile` 不同

---

## 七、要点总结

1. `volatile` 告诉编译器不要优化对该变量的访问
2. 硬件寄存器、中断标志是 `volatile` 的主要应用场景
3. `volatile` 不保证原子性，不能替代互斥锁
4. `volatile` 不提供内存屏障，不能保证指令顺序
5. C11 的 `_Atomic` 和 `<stdatomic.h>` 提供真正的原子操作
6. `volatile const` 组合用于只读硬件寄存器
7. 在 `setjmp/longjmp` 场景中，相关变量应声明为 `volatile`
8. 滥用 `volatile` 会降低程序性能
