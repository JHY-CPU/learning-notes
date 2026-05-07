# std::string转换

## 一、概念说明

C++提供了字符串与数值类型之间的双向转换。数值转字符串使用`std::to_string`（C++11），字符串转数值使用`stoi`、`stol`、`stof`等系列函数（C++11）。

## 二、具体用法

### 2.1 数值转字符串：to_string

```cpp
std::string intStr = std::to_string(42);
std::string dblStr = std::to_string(3.14159);
std::string boolStr = std::to_string(true);

std::cout << intStr << std::endl;   // 输出: 42
std::cout << dblStr << std::endl;   // 输出: 3.141590
std::cout << boolStr << std::endl;  // 输出: 1

// 拼接
std::string msg = "Age: " + std::to_string(25);
std::cout << msg << std::endl;  // 输出: Age: 25
```

### 2.2 字符串转数值：stoi系列

```cpp
std::string s1 = "42";
std::string s2 = "3.14";
std::string s3 = "0xFF";

int i = std::stoi(s1);
double d = std::stod(s2);
int hex = std::stoi(s3, nullptr, 16);  // 十六进制

std::cout << i << std::endl;    // 输出: 42
std::cout << d << std::endl;    // 输出: 3.14
std::cout << hex << std::endl;  // 输出: 255
```

### 2.3 完整函数列表

```cpp
// 字符串 → 整数
int val1 = std::stoi("123");
long val2 = std::stol("123456789");
long long val3 = std::stoll("9223372036854775807");

// 字符串 → 无符号整数
unsigned long val4 = std::stoul("4294967295");

// 字符串 → 浮点数
float val5 = std::stof("3.14");
double val6 = std::stod("3.14159265");
long double val7 = std::stold("3.14159265358979");

std::cout << val1 << " " << val6 << std::endl;
// 输出: 123 3.14159
```

### 2.4 错误处理

```cpp
try {
    int val = std::stoi("abc");  // 无效输入
} catch (const std::invalid_argument& e) {
    std::cout << "无效参数: " << e.what() << std::endl;
}

try {
    long long val = std::stoll("999999999999999999999");  // 溢出
} catch (const std::out_of_range& e) {
    std::cout << "超出范围: " << e.what() << std::endl;
}
```

## 三、注意事项与常见陷阱

- `stoi`在转换失败时抛出异常，不是返回0
- `stod`可以转换整数字符串（自动转为浮点）
- 部分转换：`stoi("42abc")`返回42，`pos`参数可获取解析位置
- `to_string`对浮点数的精度是固定的，不能控制格式
- C++20的`std::format`提供了更灵活的格式化转换
