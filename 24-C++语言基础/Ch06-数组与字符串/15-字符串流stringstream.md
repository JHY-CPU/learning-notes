# 字符串流（stringstream）

## 一、概念说明

`std::stringstream`（定义在`<sstream>`中）提供了基于字符串的流操作，可以将不同类型的数据拼接成字符串，也可以从字符串中解析出不同类型的数据。它是字符串与类型之间转换的桥梁。

## 二、具体用法

### 2.1 类型转字符串（拼接）

```cpp
#include <sstream>

std::stringstream ss;
ss << "Name: " << "Alice"
   << ", Age: " << 25
   << ", Score: " << 95.5;

std::string result = ss.str();
std::cout << result << std::endl;
// 输出: Name: Alice, Age: 25, Score: 95.5
```

### 2.2 字符串解析（提取）

```cpp
std::string data = "42 3.14 hello";
std::stringstream ss(data);

int i;
double d;
std::string s;

ss >> i >> d >> s;
std::cout << "int: " << i << std::endl;       // 输出: int: 42
std::cout << "double: " << d << std::endl;    // 输出: double: 3.14
std::cout << "string: " << s << std::endl;    // 输出: string: hello
```

### 2.3 CSV解析示例

```cpp
std::string csv = "Alice,25,95.5";
std::stringstream ss(csv);
std::string name, token;
int age;
double score;

std::getline(ss, name, ',');
ss >> age;
ss.ignore();  // 跳过逗号
ss >> score;

std::cout << name << " " << age << " " << score << std::endl;
// 输出: Alice 25 95.5
```

### 2.4 重复使用 stringstream

```cpp
std::stringstream ss;

// 构建
ss << "Count: " << 10;
std::cout << ss.str() << std::endl;  // 输出: Count: 10

// 清空重用
ss.str("");     // 清空内容
ss.clear();     // 清空状态标志

ss << "New: " << 20;
std::cout << ss.str() << std::endl;  // 输出: New: 20
```

### 2.5 ostringstream 和 istringstream

```cpp
// 只用于输出
std::ostringstream oss;
oss << std::fixed << std::setprecision(2) << 3.14159;
std::cout << oss.str() << std::endl;  // 输出: 3.14

// 只用于输入
std::istringstream iss("100 200 300");
int a, b, c;
iss >> a >> b >> c;
std::cout << a + b + c << std::endl;  // 输出: 600
```

## 三、注意事项与常见陷阱

- `str()`返回内容的拷贝，频繁调用有性能开销
- 解析失败后流进入错误状态，需`clear()`重置
- `str("")`清空内容，`clear()`清空状态，两者需配合使用
- `stringstream`默认使用空格分隔，`getline`可指定分隔符
- 大量格式化操作考虑使用`std::format`（C++20）
