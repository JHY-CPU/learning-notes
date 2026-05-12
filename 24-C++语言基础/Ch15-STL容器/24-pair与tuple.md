# pair与tuple

## 一、概念说明

`std::pair`存储两个值（C++标准 §20.4.2），`std::tuple`存储任意数量的值（C++标准 §20.5.2）。它们常用于从函数返回多个值，以及作为map的元素类型（`map::value_type`就是`pair<const Key, Value>`）。C++17的结构化绑定使它们的使用更加简洁。

### 1.1 核心对比

| 特性 | pair | tuple |
|------|------|-------|
| 元素数量 | 固定2个 | 任意数量 |
| 访问 | `first`/`second` | `get<I>()` |
| 结构化绑定 | 支持 | 支持 |
| 比较 | 支持 | 支持 |
| 创建 | `make_pair` | `make_tuple` |

## 二、具体用法

### 2.1 std::pair

```cpp
#include <utility>
#include <iostream>
#include <string>

int main() {
    // 创建方式
    std::pair<std::string, int> p1("age", 25);
    auto p2 = std::make_pair("score", 90);           // C++11
    auto p3 = std::pair{"name", std::string("Bob")}; // C++17推导

    // 访问
    std::cout << p1.first << ": " << p1.second << std::endl;  // age: 25

    // 结构化绑定（C++17，推荐）
    auto [key, value] = p2;
    std::cout << key << ": " << value << std::endl;  // score: 90

    // 比较（先比first再比second）
    std::pair<int, int> a{1, 2}, b{1, 3};
    std::cout << (a < b) << std::endl;   // true
    std::cout << (a == b) << std::endl;  // false

    // 修改
    p1.second = 30;

    // 用于map
    std::map<std::string, int> m;
    m.insert(std::make_pair("key", 42));
}
```

### 2.2 std::tuple

```cpp
#include <tuple>
#include <string>
#include <iostream>

void tuple_demo() {
    // 创建
    auto t1 = std::make_tuple(1, 3.14, std::string("hello"));
    std::tuple<int, double, std::string> t2(42, 2.72, "world");
    auto t3 = std::tuple{1, 2.0, 'a'};  // C++17推导

    // 索引访问（编译期）
    std::cout << std::get<0>(t1) << std::endl;  // 1
    std::cout << std::get<2>(t1) << std::endl;  // hello

    // 按类型访问（同类型不能重复出现）
    std::tuple<int, double, char> t4(1, 2.0, 'a');
    std::cout << std::get<double>(t4) << std::endl;  // 2.0

    // C++14: 按类型获取引用
    std::get<int>(t4) = 100;

    // 结构化绑定（C++17，推荐）
    auto [a, b, c] = t1;
    std::cout << a << " " << b << " " << c << std::endl;

    // tie解包（C++11）
    int x; double y; std::string z;
    std::tie(x, y, z) = t2;

    // 忽略某些值
    std::tie(x, std::ignore, z) = t2;  // 忽略第二个元素

    // 编译期信息
    std::cout << "size: " << std::tuple_size_v<decltype(t1)> << std::endl;  // 3
}
```

### 2.3 tuple_cat与高级操作

```cpp
#include <tuple>

void tuple_advanced() {
    auto t1 = std::make_tuple(1, 2);
    auto t2 = std::make_tuple(3.0, "hello");

    // 拼接
    auto t3 = std::tuple_cat(t1, t2);  // (1, 2, 3.0, "hello")

    // 比较（逐元素比较）
    auto a = std::make_tuple(1, 2, 3);
    auto b = std::make_tuple(1, 2, 4);
    std::cout << (a < b) << std::endl;  // true（3 < 4）

    // 用于多返回值
    auto divide = [](int a, int b) -> std::tuple<int, int> {
        return {a / b, a % b};
    };
    auto [quotient, remainder] = divide(17, 5);
    std::cout << "商: " << quotient << " 余数: " << remainder << std::endl;
}
```

### 2.4 apply调用函数（C++17）

```cpp
#include <tuple>

void apply_demo() {
    auto add = [](int a, int b, int c) { return a + b + c; };
    auto args = std::make_tuple(1, 2, 3);

    // 将tuple展开为函数参数
    int result = std::apply(add, args);  // 6

    // 实用场景：配置参数
    auto configure = [](int width, int height, bool fullscreen) {
        std::cout << width << "x" << height
                  << (fullscreen ? " 全屏" : " 窗口") << std::endl;
    };
    auto settings = std::make_tuple(1920, 1080, true);
    std::apply(configure, settings);
}
```

## 三、注意事项与常见陷阱

1. **pair用于固定两个值，tuple用于多个值**：pair更简洁
2. **`std::get<I>`索引必须是编译期常量**：不能用变量
3. **结构化绑定是C++17最简洁的方式**：`auto [a, b, c] = tuple;`
4. **`std::tie`可以忽略某些值**：`std::tie(a, std::ignore, c) = tuple`
5. **map的`value_type`是`pair<const Key, Value>`**：键是const
6. **tuple的比较是逐元素的**：从第一个元素开始比较
7. **`std::apply`（C++17）**：将tuple展开为函数参数
