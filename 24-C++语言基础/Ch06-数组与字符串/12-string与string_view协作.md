# string与string_view协作

## 一、概念说明

`std::string`和`std::string_view`各有适用场景：`string`拥有数据所有权，`string_view`提供高效只读访问。合理选择参数传递策略是编写高效C++代码的关键。

## 二、具体用法

### 2.1 参数传递策略

```cpp
// 规则：
// 1. 需要拷贝/修改/存储 → const std::string&
// 2. 只读查看 → std::string_view
// 3. 接受所有权 → std::string (按值) + std::move

void readOnly(std::string_view sv) {
    std::cout << sv << std::endl;
}

void takeOwnership(std::string s) {  // 按值接收
    std::cout << s << std::endl;
}

void modify(std::string& s) {  // 非const引用
    s += " modified";
}
```

### 2.2 string_view转string

```cpp
std::string_view sv = "Hello";

// 显式转换（有拷贝）
std::string str(sv);
std::string str2(sv.data(), sv.size());

// 需要修改时必须转换
std::string mutableStr(sv);
mutableStr += " World";
std::cout << mutableStr << std::endl;  // 输出: Hello World
```

### 2.3 string转string_view

```cpp
std::string str = "Hello World";

// 隐式转换（无拷贝）
std::string_view sv = str;  // OK

// 也可以从子串创建
std::string_view sub = std::string_view(str).substr(6, 5);
std::cout << sub << std::endl;  // 输出: World
```

### 2.4 生命周期注意事项

```cpp
// 危险：返回局部string的string_view
std::string_view bad() {
    std::string temp = "danger";
    return temp;  // 悬垂引用！temp析构后sv无效
}

// 正确：返回string
std::string good() {
    return "safe";  // 有所有权
}
```

### 2.5 混合使用

```cpp
std::string name = "Alice";

// string_view不会延长临时对象的生命周期
auto sv = std::string_view("Hello " + name);  // 临时string，危险！
// 正确做法
std::string full = "Hello " + name;
auto sv2 = std::string_view(full);  // OK，full还活着
```

## 三、注意事项与常见陷阱

- `string_view`不会延长临时对象生命周期
- 不要将`string_view`存入容器或类成员（除非底层数据始终有效）
- 函数返回值如果是新字符串，返回`string`而非`string_view`
- `string_view`可以高效遍历大字符串的子串
- C++17之前用`const std::string&`，C++17后函数参数优先`string_view`
