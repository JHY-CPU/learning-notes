# cin详解

## 一、概念说明

`std::cin`是标准输入流对象，类型为`std::istream`。输入运算符`>>`自动跳过空白字符（空格、制表符、换行），并根据变量类型进行解析。

`cin`遇到输入结束（EOF或类型不匹配）时会设置失败状态，后续读取全部失败。

```cpp
#include <iostream>
#include <string>

int main() {
    int age;
    double height;
    std::string name;

    std::cout << "请输入姓名: ";
    std::cin >> name; // >> 跳过前导空白，遇到空白停止

    std::cout << "请输入年龄: ";
    std::cin >> age;

    std::cout << "请输入身高: ";
    std::cin >> height;

    std::cout << name << ", " << age << "岁, " << height << "米" << std::endl;

    return 0;
}
```

**输出（示例）：**
```
请输入姓名: 张三
请输入年龄: 25
请输入身高: 1.75
张三, 25岁, 1.75米
```

## 二、具体用法

### 2.1 读取整行

```cpp
#include <iostream>
#include <string>

int main() {
    std::string fullName;

    std::cout << "请输入全名: ";
    std::cin.ignore(); // 清除之前>>残留的换行符
    std::getline(std::cin, fullName); // 读取整行（含空格）

    std::cout << "你好, " << fullName << std::endl;

    // 读取多行
    std::cout << "输入多行（输入空行结束）:" << std::endl;
    std::string line;
    while (std::getline(std::cin, line) && !line.empty()) {
        std::cout << ">> " << line << std::endl;
    }

    return 0;
}
```

**输出（示例）：**
```
请输入全名: 张三 李四
你好, 张三 李四
输入多行（输入空行结束）:
第一行
>> 第一行
第二行
>> 第二行

```

### 2.2 输入验证与错误恢复

```cpp
#include <iostream>
#include <limits>

int main() {
    int number;

    std::cout << "请输入一个整数: ";
    while (!(std::cin >> number)) {
        std::cin.clear(); // 清除错误状态
        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        std::cout << "输入无效，请重新输入: ";
    }

    std::cout << "你输入的是: " << number << std::endl;
    return 0;
}
```

**输出（示例）：**
```
请输入一个整数: abc
输入无效，请重新输入: 42
你输入的是: 42
```

## 三、注意事项与常见陷阱

- **`>>`跳过空白**：无法读取含空格的字符串，用`getline`代替。
- **`getline`读取`>>`残留的换行符**：混合使用前先`cin.ignore()`。
- **`cin >>`失败后流进入fail状态**：必须`clear()`才能继续使用。
- **`peek()`查看下一个字符而不消费**：可用于预判输入类型。
- **`cin.sync_with_stdio(false)`可加速**：但不能混用C的`scanf`。
