# C++17 optional variant any

## 一、概念说明

C++17引入三种类型安全的值语义容器（C++17 §23.6, §23.7）：
- `std::optional<T>`：可能有值也可能没有（替代指针/特殊值作可选值）
- `std::variant<T1, T2, ...>`：类型安全的联合体（替代`union`）
- `std::any`：可以持有任意类型（类型擦除）

### 1.1 三者对比

| 特性 | optional | variant | any |
|------|----------|---------|-----|
| 类型数量 | 1个类型+空 | 固定N个类型 | 任意类型 |
| 类型安全 | 是 | 是 | 运行时检查 |
| 内存开销 | 值+bool | 最大类型+index | 堆分配（大对象） |
| 典型用例 | 可能失败的返回值 | 状态机、多态值 | 配置存储 |

```cpp
#include <iostream>
#include <optional>
#include <variant>
#include <any>
#include <string>

// optional：可能失败的函数
std::optional<int> safeDivide(int a, int b) {
    if (b == 0) return std::nullopt;
    return a / b;
}

int main() {
    // optional
    if (auto result = safeDivide(10, 3)) {
        std::cout << "10/3=" << *result << std::endl;
    }
    if (auto result = safeDivide(10, 0); !result) {
        std::cout << "除零失败" << std::endl;
    }

    // variant
    std::variant<int, double, std::string> v = 42;
    std::cout << "int: " << std::get<int>(v) << std::endl;
    v = 3.14;
    std::cout << "double: " << std::get<double>(v) << std::endl;
    v = "hello";
    std::cout << "string: " << std::get<std::string>(v) << std::endl;

    // any
    std::any a = 42;
    std::cout << "any int: " << std::any_cast<int>(a) << std::endl;
    a = std::string("world");
    std::cout << "any string: " << std::any_cast<std::string>(a) << std::endl;

    return 0;
}
```

**输出：**
```
10/3=3
除零失败
int: 42
double: 3.14
string: hello
any int: 42
any string: world
```

## 二、具体用法

### 2.1 optional进阶

```cpp
#include <iostream>
#include <optional>
#include <string>
#include <fstream>

// 配置读取
std::optional<std::string> readConfig(const std::string& key) {
    static std::map<std::string, std::string> config = {
        {"host", "localhost"}, {"port", "8080"}
    };
    auto it = config.find(key);
    if (it != config.end()) return it->second;
    return std::nullopt;
}

// 链式操作
std::optional<int> parseInt(const std::string& s) {
    try { return std::stoi(s); }
    catch (...) { return std::nullopt; }
}

int main() {
    // value_or提供默认值
    auto host = readConfig("host").value_or("127.0.0.1");
    auto timeout = readConfig("timeout").value_or("30");
    std::cout << "host=" << host << ", timeout=" << timeout << std::endl;

    // emplace直接构造
    std::optional<std::string> opt;
    opt.emplace("hello");
    std::cout << "opt=" << *opt << std::endl;

    // 函数式操作
    auto result = parseInt("42");
    if (result.has_value()) {
        std::cout << "parsed: " << result.value() << std::endl;
    }

    return 0;
}
```

### 2.2 variant访问

```cpp
#include <iostream>
#include <variant>
#include <string>

struct Visitor {
    void operator()(int i) const { std::cout << "int: " << i << std::endl; }
    void operator()(double d) const { std::cout << "double: " << d << std::endl; }
    void operator()(const std::string& s) const { std::cout << "string: " << s << std::endl; }
};

// overloaded模式（C++17常见惯用法）
template<class... Ts> struct overloaded : Ts... { using Ts::operator()...; };
template<class... Ts> overloaded(Ts...) -> overloaded<Ts...>;

int main() {
    std::variant<int, double, std::string> v = 3.14;

    // std::visit + 访问者
    std::visit(Visitor{}, v);

    // lambda + overloaded
    std::visit(overloaded{
        [](int i) { std::cout << "lambda int: " << i << std::endl; },
        [](double d) { std::cout << "lambda double: " << d << std::endl; },
        [](const std::string& s) { std::cout << "lambda string: " << s << std::endl; }
    }, v);

    // 状态机
    struct State {
        std::variant<std::monostate, int, std::string> data;
    };

    return 0;
}
```

### 2.3 any使用

```cpp
#include <iostream>
#include <any>
#include <string>
#include <map>

// 类型擦除的属性存储
class Properties {
    std::map<std::string, std::any> data;
public:
    template <typename T>
    void set(const std::string& key, T value) {
        data[key] = std::move(value);
    }

    template <typename T>
    std::optional<T> get(const std::string& key) const {
        auto it = data.find(key);
        if (it != data.end()) {
            try { return std::any_cast<T>(it->second); }
            catch (...) { return std::nullopt; }
        }
        return std::nullopt;
    }
};

int main() {
    Properties props;
    props.set("name", std::string("Alice"));
    props.set("age", 30);
    props.set("score", 95.5);

    if (auto name = props.get<std::string>("name"))
        std::cout << "name=" << *name << std::endl;
    if (auto age = props.get<int>("age"))
        std::cout << "age=" << *age << std::endl;

    // 类型检查
    std::any a = 42;
    std::cout << "type: " << a.type().name() << std::endl;
    std::cout << "has value: " << a.has_value() << std::endl;

    return 0;
}
```

## 三、注意事项与常见陷阱

1. **`optional`的`value()`在无值时抛`bad_optional_access`**：用`value_or()`提供默认值。
2. **`variant`的第一个类型必须可默认构造**：使用`std::monostate`占位。
3. **`std::get`类型错误抛`bad_variant_access`**：用`std::get_if`安全检查。
4. **`any`使用`type()`检查类型**：`any_cast`失败抛`bad_any_cast`。
5. **`optional`/`variant`比`unique_ptr`/`any`更高效**：不需要堆分配（小对象）。
6. **`variant`不支持引用类型**：用`std::reference_wrapper`包装。
