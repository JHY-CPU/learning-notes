# JSON处理

## 一、概念说明

C++标准库没有内置JSON支持，常用第三方库`nlohmann/json`（单头文件库）。它提供了直观的JSON创建、解析、序列化接口，支持STL容器风格的操作。

安装：将`json.hpp`放到项目中，或通过vcpkg/CMake FetchContent安装。

```cpp
#include <iostream>
#include <fstream>
// #include <nlohmann/json.hpp>  // 实际使用时取消注释
// using json = nlohmann::json;

int main() {
    // 以下为概念性代码（需安装nlohmann/json）
    /*
    // 创建JSON对象
    json person;
    person["name"] = "张三";
    person["age"] = 25;
    person["skills"] = {"C++", "Python", "Java"};
    person["address"]["city"] = "北京";
    person["address"]["zip"] = "100000";

    // 序列化为字符串
    std::string str = person.dump(4); // 缩进4空格
    std::cout << str << std::endl;

    // 写入文件
    std::ofstream("person.json") << str;

    // 从文件读取
    std::ifstream ifs("person.json");
    json loaded = json::parse(ifs);
    std::cout << "姓名: " << loaded["name"] << std::endl;
    */

    std::cout << "nlohmann/json 使用示例:" << std::endl;
    std::cout << "json obj;" << std::endl;
    std::cout << R"(obj["key"] = "value";)" << std::endl;
    std::cout << R"(obj["array"] = {1, 2, 3};)" << std::endl;
    std::cout << "std::string s = obj.dump();" << std::endl;

    return 0;
}
```

**输出：**
```
nlohmann/json 使用示例:
json obj;
obj["key"] = "value";
obj["array"] = {1, 2, 3};
std::string s = obj.dump();
```

## 二、具体用法

### 2.1 nlohmann/json常用API

```cpp
// 安装: vcpkg install nlohmann-json 或在CMake中 FetchContent
// #include <nlohmann/json.hpp>
// using json = nlohmann::json;

// 创建
json j;
j["int"] = 42;
j["double"] = 3.14;
j["bool"] = true;
j["string"] = "hello";
j["null"] = nullptr;
j["array"] = {1, 2, 3};
j["object"] = {{"key", "value"}};

// 读取
int i = j["int"];                    // 42
std::string s = j["string"];         // "hello"
auto arr = j["array"].get<std::vector<int>>(); // {1,2,3}

// 检查存在
if (j.contains("key")) { /* ... */ }
if (j["key"].is_string()) { /* ... */ }

// 遍历
for (auto& [key, value] : j.items()) {
    std::cout << key << ": " << value << std::endl;
}

// 序列化/反序列化
std::string json_str = j.dump(4);    // 格式化输出
json parsed = json::parse(json_str); // 解析字符串

// 异常处理
try {
    json::parse("invalid json");
} catch (json::parse_error& e) {
    std::cout << e.what() << std::endl;
}
```

### 2.2 与STL容器互转

```cpp
// vector <-> json array
std::vector<int> vec = {1, 2, 3};
json j_arr = vec;  // 自动转为JSON数组
auto vec2 = j_arr.get<std::vector<int>>();

// map <-> json object
std::map<std::string, int> m = {{"a", 1}, {"b", 2}};
json j_obj = m;
auto m2 = j_obj.get<std::map<std::string, int>>();

// struct <-> json (自定义转换)
struct Person {
    std::string name;
    int age;
};
void to_json(json& j, const Person& p) {
    j = json{{"name", p.name}, {"age", p.age}};
}
void from_json(const json& j, Person& p) {
    j.at("name").get_to(p.name);
    j.at("age").get_to(p.age);
}
```

## 三、注意事项与常见陷阱

- **`nlohmann/json`是单头文件库**：包含即用，但编译较慢。
- **异常处理很重要**：`at()`访问不存在的key会抛异常。
- **`operator[]`会自动创建key**：读取时用`value()`或`at()`更安全。
- **大文件考虑流式解析**：`json::parse`会一次性加载到内存。
- **自定义类型需实现`to_json`/`from_json`函数**：在相同命名空间中。
