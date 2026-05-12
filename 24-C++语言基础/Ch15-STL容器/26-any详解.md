# any详解

## 一、概念说明

`std::any`是C++17引入的类型擦除容器（C++标准 §20.8.4），可以存储任意类型的单个值。它提供了运行时的多态存储，但需要类型检查和转换。相比`variant`，any不需要编译时确定所有可能类型，但访问开销更大。

### 1.1 any vs variant vs void*

| 特性 | any | variant | void* |
|------|-----|---------|-------|
| 类型安全 | 是 | 是 | 否 |
| 编译时类型检查 | 否 | 是 | 否 |
| 性能 | 中 | 高 | 最高 |
| 存储类型数 | 任意 | 编译时确定 | 任意 |
| 引入版本 | C++17 | C++17 | C++98 |

## 二、具体用法

### 2.1 基本操作

```cpp
#include <any>
#include <iostream>
#include <string>
#include <vector>

int main() {
    // 存储不同类型
    std::any a = 42;
    std::cout << a.type().name() << std::endl;  // int的类型名

    a = 3.14;                     // 现在是double
    a = std::string("hello");     // 现在是string

    // 安全获取（异常版本）
    try {
        auto s = std::any_cast<std::string>(a);
        std::cout << s << std::endl;  // hello
    } catch (const std::bad_any_cast& e) {
        std::cerr << e.what() << std::endl;
    }

    // 指针版本（失败返回nullptr，推荐）
    auto* p = std::any_cast<int>(&a);
    if (p) std::cout << *p << std::endl;
    else std::cout << "不是int" << std::endl;

    // 检查和重置
    std::cout << "has_value: " << a.has_value() << std::endl;  // true
    a.reset();
    std::cout << "has_value: " << a.has_value() << std::endl;  // false
}
```

### 2.2 emplace与in_place

```cpp
void construct_demo() {
    // emplace直接构造
    std::any a;
    a.emplace<std::string>("hello");

    // in_place构造
    std::any b(std::in_place_type<std::vector<int>>, {1, 2, 3});

    // 拷贝和移动
    std::any c = a;              // 拷贝
    std::any d = std::move(a);   // 移动（a变空）
}
```

### 2.3 类型检查

```cpp
#include <typeinfo>

void type_check() {
    std::any a = 42;

    // 检查类型
    if (a.type() == typeid(int))
        std::cout << "是int" << std::endl;

    if (a.type() == typeid(std::string))
        std::cout << "是string" << std::endl;

    // 实际类型名（实现相关）
    std::cout << a.type().name() << std::endl;
}
```

### 2.4 实用示例：通用属性存储

```cpp
#include <map>
#include <any>

class Properties {
    std::map<std::string, std::any> data;
public:
    template<typename T>
    void set(const std::string& key, T&& value) {
        data[key] = std::forward<T>(value);
    }

    template<typename T>
    std::optional<T> get(const std::string& key) const {
        auto it = data.find(key);
        if (it == data.end()) return std::nullopt;
        return std::any_cast<T>(it->second);
    }

    void demo() {
        set("name", std::string("Alice"));
        set("age", 25);
        set("score", 95.5);

        auto name = get<std::string>("name");
        auto age = get<int>("age");
        if (name) std::cout << *name << std::endl;
    }
};
```

## 三、注意事项与常见陷阱

1. **any有小对象优化（SSO）**：小值存在栈上，大对象可能有堆分配
2. **`any_cast<T>`类型必须精确匹配**：`int`和`long`不同
3. **性能开销高于variant**：运行时类型检查+可能的堆分配
4. **适用于需要存储未知类型的场景**：插件系统、配置、脚本绑定
5. **any不能存储引用**：用`reference_wrapper`包装
6. **`reset()`释放存储的值**：调用析构函数
7. **拷贝any会拷贝存储的值**：深拷贝语义
