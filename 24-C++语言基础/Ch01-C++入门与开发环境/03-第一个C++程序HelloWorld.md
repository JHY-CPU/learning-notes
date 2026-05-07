# 第一个C++程序HelloWorld

## 一、概念说明

每个编程语言的学习都从"Hello, World!"开始。这个简单的程序涵盖了C++程序的基本组成部分：预处理指令、主函数、输入输出流和返回值。

## 二、具体用法

### 2.1 最简单的HelloWorld程序

```cpp
#include <iostream>  // 预处理指令：包含输入输出库

int main() {          // 主函数：程序入口
    std::cout << "Hello, World!" << std::endl;  // 输出语句
    return 0;         // 返回状态码，0表示成功
}
```

输出：
```
Hello, World!
```

### 2.2 逐行解析

```cpp
// 第1行：预处理指令
#include <iostream>
// #include 告诉编译器在编译前将指定头文件的内容插入到此处
// <iostream> 是C++标准输入输出流库，包含cout、cin等

// 第3行：主函数声明
int main()
// int 表示函数返回整型值
// main 是程序的入口函数，每个C++程序必须有且仅有一个main函数
// () 内可以不写参数，也可以写 int argc, char* argv[]

// 第4行：输出语句
std::cout << "Hello, World!" << std::endl;
// std::cout 标准输出流对象（控制台输出）
// << 是流插入运算符，将右侧数据插入到输出流
// std::endl 输出换行并刷新缓冲区

// 第5行：返回语句
return 0;
// 向操作系统返回状态码，0通常表示程序正常结束
```

### 2.3 使用using namespace简化

```cpp
#include <iostream>
using namespace std;  // 使用std命名空间，避免每次写std::

int main() {
    cout << "Hello, World!" << endl;
    cout << "你好，C++！" << endl;
    return 0;
}
```

输出：
```
Hello, World!
你好，C++！
```

### 2.4 带命令行参数的版本

```cpp
#include <iostream>
using namespace std;

int main(int argc, char* argv[]) {
    cout << "程序名: " << argv[0] << endl;
    cout << "参数个数: " << argc << endl;

    for (int i = 1; i < argc; i++) {
        cout << "参数" << i << ": " << argv[i] << endl;
    }
    return 0;
}
```

运行命令及输出：
```
$ ./hello Alice Bob
程序名: ./hello
参数个数: 3
参数1: Alice
参数2: Bob
```

### 2.5 编译与运行

```bash
# 使用g++编译
g++ -std=c++17 -o hello hello.cpp

# 运行程序
./hello

# 一步完成编译运行
g++ -std=c++17 hello.cpp -o hello && ./hello
```

## 三、注意事项与常见陷阱

1. **main函数必须返回int**：C++标准要求main函数的返回类型为`int`，不能省略
2. **分号不能少**：每条C++语句以分号`;`结尾，忘记分号是最常见的编译错误
3. **大小写敏感**：`cout`和`Cout`是不同的标识符，注意大小写
4. **头文件包含**：使用`cout`必须包含`<iostream>`，否则编译报错
5. **不要在头文件后加分号**：`#include <iostream>;`是错误的写法
6. **endl vs \n**：`endl`会刷新缓冲区效率较低，大量输出时建议用`'\n'`
