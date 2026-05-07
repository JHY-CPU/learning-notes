# cout详解

## 一、概念说明

`std::cout`是标准输出流对象，类型为`std::ostream`。输出运算符`<<`支持链式调用，自动根据类型格式化输出。

缓冲机制：
- **全缓冲**：缓冲区满时刷新（如文件流）
- **行缓冲**：遇到换行时刷新（如`cout`）
- **无缓冲**：立即输出（如`cerr`）

```cpp
#include <iostream>

int main() {
    // 链式输出
    std::cout << "Hello" << " " << "World" << "!" << std::endl;

    // 各种类型自动格式化
    std::cout << "整数: " << 42 << std::endl;
    std::cout << "浮点: " << 3.14159 << std::endl;
    std::cout << "字符: " << 'A' << std::endl;
    std::cout << "布尔: " << true << " " << false << std::endl;
    std::cout << "指针: " << (void*)"test" << std::endl;

    // endl vs '\n'
    std::cout << "带endl（刷新缓冲区）" << std::endl;
    std::cout << "带\\n（不刷新缓冲区）" << '\n';

    // 手动刷新
    std::cout << "手动刷新" << std::flush;

    return 0;
}
```

**输出：**
```
Hello World!
整数: 42
浮点: 3.14159
字符: A
布尔: 1 0
指针: 0x...
带endl（刷新缓冲区）
带\n（不刷新缓冲区）
手动刷新
```

## 二、具体用法

### 2.1 输出控制

```cpp
#include <iostream>

int main() {
    // boolalpha: 输出true/false而非1/0
    std::cout << std::boolalpha << true << " " << false << std::endl;

    // hex/oct/dec: 进制切换
    std::cout << std::hex << 255 << std::endl;    // ff
    std::cout << std::oct << 255 << std::endl;    // 377
    std::cout << std::dec << 255 << std::endl;    // 255

    // showbase: 显示进制前缀
    std::cout << std::showbase << std::hex << 255 << std::endl; // 0xff

    // 左右对齐与填充
    std::cout << "|" << std::setw(10) << "Hi" << "|" << std::endl;
    std::cout << "|" << std::left << std::setw(10) << "Hi" << "|" << std::endl;

    return 0;
}
```

**输出：**
```
true false
ff
377
255
0xff
|        Hi|
|Hi        |
```

### 2.2 缓冲区管理

```cpp
#include <iostream>
#include <thread>
#include <chrono>

int main() {
    // 无缓冲输出（cerr）
    std::cerr << "错误: 立即输出";
    // cerr不加endl也会立即输出

    // 有缓冲输出
    std::cout << "缓冲输出";
    // 不刷新的话可能不会立即显示
    std::cout << std::endl; // endl = '\n' + flush

    // unitbuf: 每次输出后自动刷新
    std::cout << std::unitbuf << "自动刷新1" << "自动刷新2";
    std::cout << std::nounitbuf; // 关闭自动刷新

    return 0;
}
```

**输出：**
```
错误: 立即输出缓冲输出
自动刷新1自动刷新2
```

## 三、注意事项与常见陷阱

- **`endl`性能开销大**：循环中用`'\n'`代替，最后再`flush`。
- **多线程中`cout`不是线程安全的**：需自行加锁或用原子写入。
- **`cout`和`printf`混用可能乱序**：因为缓冲区不同。
- **`sync_with_stdio(false)`关闭同步后不能混用C/C++ IO**。
- **`cerr`不缓冲但可重定向**：`2>error.txt`可重定向标准错误。
