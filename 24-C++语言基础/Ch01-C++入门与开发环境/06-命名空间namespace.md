# 命名空间 namespace

## 一、概念说明

命名空间（namespace）用于组织代码、避免**命名冲突**。当多个库可能定义同名函数或类时，命名空间提供了隔离机制。C++标准库的所有内容都在`std`命名空间中。

## 二、具体用法

### 2.1 定义命名空间

```cpp
#include <iostream>

// 定义自己的命名空间
namespace MyLib {
    int version = 1;

    void print() {
        std::cout << "来自MyLib的print函数" << std::endl;
    }

    class Calculator {
    public:
        int add(int a, int b) { return a + b; }
    };
}

int main() {
    std::cout << "版本: " << MyLib::version << std::endl;
    MyLib::print();

    MyLib::Calculator calc;
    std::cout << "3 + 4 = " << calc.add(3, 4) << std::endl;
    return 0;
}
```

输出：
```
版本: 1
来自MyLib的print函数
3 + 4 = 7
```

### 2.2 using 声明和指令

```cpp
#include <iostream>
#include <string>
#include <vector>

// 方式一：using声明 —— 引入单个名称（推荐）
using std::cout;
using std::endl;
using std::string;

// 方式二：using指令 —— 引入整个命名空间（头文件中避免使用）
// using namespace std;

namespace Physics {
    double gravity = 9.8;
    double calculateForce(double mass) {
        return mass * gravity;
    }
}

// using声明引入特定名称
using Physics::gravity;

int main() {
    string name = "C++";
    cout << name << endl;
    cout << "重力加速度: " << gravity << endl;
    cout << "10kg物体受力: " << Physics::calculateForce(10) << "N" << endl;
    return 0;
}
```

输出：
```
C++
重力加速度: 9.8
10kg物体受力: 98N
```

### 2.3 嵌套命名空间

```cpp
#include <iostream>

// 嵌套命名空间
namespace Company {
    namespace Project {
        namespace Utils {
            void log(const char* msg) {
                std::cout << "[LOG] " << msg << std::endl;
            }
        }
    }
}

// C++17简化写法
namespace Company::Project::Config {
    constexpr int MAX_CONNECTIONS = 100;
}

int main() {
    Company::Project::Utils::log("系统启动");
    std::cout << "最大连接数: " << Company::Project::Config::MAX_CONNECTIONS << std::endl;
    return 0;
}
```

输出：
```
[LOG] 系统启动
最大连接数: 100
```

### 2.4 匿名命名空间

```cpp
#include <iostream>

// 匿名命名空间：其中的标识符仅在当前文件可见（类似static）
namespace {
    int internalCounter = 0;

    void helperFunction() {
        internalCounter++;
        std::cout << "调用次数: " << internalCounter << std::endl;
    }
}

int main() {
    helperFunction();
    helperFunction();
    helperFunction();
    return 0;
}
```

输出：
```
调用次数: 1
调用次数: 2
调用次数: 3
```

### 2.5 命名空间别名

```cpp
#include <iostream>

namespace VeryLongNamespaceName {
    void func() { std::cout << "长命名空间的函数" << std::endl; }
}

// 创建别名
namespace Alias = VeryLongNamespaceName;

int main() {
    Alias::func();  // 比写全名简洁
    return 0;
}
```

输出：
```
长命名空间的函数
```

## 三、注意事项与常见陷阱

1. **不要在头文件中使用 `using namespace std`**：这会污染所有包含该头文件的源文件的命名空间
2. **命名空间可以分段定义**：同名命名空间的多个定义会自动合并
3. **匿名命名空间替代static**：C++中匿名命名空间是比`static`更推荐的文件作用域限制方式
4. **命名空间不以分号结尾**：`namespace A { }`后面不需要分号
5. **using声明的作用域**：`using`声明出现在哪个作用域，就只在该作用域内有效
