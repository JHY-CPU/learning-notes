# main函数详解

## 一、概念说明

`main`函数是C++程序的**入口点**。操作系统调用`main`启动程序，程序执行完`main`后终止。`main`函数有特殊规则：不能重载、不能被其他函数调用、不能声明为`constexpr`/`inline`/`static`。

## 二、具体用法

### 2.1 基本形式

```cpp
// 无参数形式
int main() {
    std::cout << "Hello, World!" << std::endl;
    return 0;  // 返回0表示成功
}

// 带参数形式（命令行参数）
int main(int argc, char* argv[]) {
    // argc: 参数个数（包括程序名）
    // argv: 参数字符串数组
    std::cout << "程序名: " << argv[0] << std::endl;
    std::cout << "参数个数: " << argc << std::endl;

    for (int i = 0; i < argc; ++i) {
        std::cout << "argv[" << i << "] = " << argv[i] << std::endl;
    }
    return 0;
}
```

```
// 运行: ./myprogram hello world 42
// 输出:
// 程序名: ./myprogram
// 参数个数: 4
// argv[0] = ./myprogram
// argv[1] = hello
// argv[2] = world
// argv[3] = 42
```

### 2.2 main的返回值

```cpp
int main() {
    // return 0 或 return EXIT_SUCCESS 表示成功
    // return 1 或 return EXIT_FAILURE 表示失败
    #include <cstdlib>

    bool success = doWork();
    return success ? EXIT_SUCCESS : EXIT_FAILURE;
}

// C++允许省略return，默认返回0（C++11起）
int main() {
    // 隐式 return 0;
}
```

### 2.3 main之前和之后的执行

```cpp
// 全局对象的构造函数在main之前执行
struct Init {
    Init() { std::cout << "main之前\n"; }
    ~Init() { std::cout << "main之后\n"; }
};

Init globalInit;

int main() {
    std::cout << "main执行中\n";
    return 0;
}
// 输出:
// main之前
// main执行中
// main之后
```

## 三、注意事项与常见陷阱

- `main`只能返回`int`类型
- `main`不能被重载，不能有多个main函数
- 命令行参数`argv[argc]`保证为`nullptr`
- 跨平台使用`main`参数时注意编码问题（Windows宽字符版本`wmain`）
- `main`中未捕获的异常会导致`std::terminate`调用
