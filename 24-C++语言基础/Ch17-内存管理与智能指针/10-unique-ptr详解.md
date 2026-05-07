# unique_ptr详解

## 一、概念说明

`std::unique_ptr`是C++11引入的独占所有权智能指针，同一时刻只能有一个`unique_ptr`拥有某块内存。它不支持拷贝，但支持移动，大小与裸指针相同（零开销抽象）。

使用`std::make_unique<T>(args...)`（C++14）创建是推荐方式，避免直接使用`new`。

```cpp
#include <iostream>
#include <memory>

class Resource {
    std::string name;
public:
    Resource(std::string n) : name(std::move(n)) {
        std::cout << "Resource '" << name << "' 创建" << std::endl;
    }
    ~Resource() {
        std::cout << "Resource '" << name << "' 销毁" << std::endl;
    }
    void use() const { std::cout << "使用: " << name << std::endl; }
};

int main() {
    // 推荐：make_unique
    auto p1 = std::make_unique<Resource>("数据库连接");
    p1->use();

    // 移动所有权
    auto p2 = std::move(p1);
    std::cout << "p1 is null: " << (p1 == nullptr) << std::endl;
    p2->use();

    // 离开作用域自动释放
    return 0;
}
```

**输出：**
```
Resource '数据库连接' 创建
使用: 数据库连接
p1 is null: 1
使用: 数据库连接
Resource '数据库连接' 销毁
```

## 二、具体用法

### 2.1 基本操作

```cpp
#include <iostream>
#include <memory>

int main() {
    auto ptr = std::make_unique<int>(42);

    // 解引用
    std::cout << "值: " << *ptr << std::endl;

    // 获取裸指针（不转移所有权）
    int* raw = ptr.get();
    std::cout << "裸指针值: " << *raw << std::endl;

    // 重置
    ptr.reset(new int(100));
    std::cout << "重置后: " << *ptr << std::endl;

    // 释放所有权（返回裸指针，需手动管理）
    int* released = ptr.release();
    std::cout << "释放后: " << *released << std::endl;
    std::cout << "ptr is null: " << (ptr == nullptr) << std::endl;
    delete released;

    return 0;
}
```

**输出：**
```
值: 42
裸指针值: 42
重置后: 100
释放后: 100
ptr is null: 1
```

### 2.2 作为函数返回值

```cpp
#include <iostream>
#include <memory>

class Config {
    int value;
public:
    Config(int v) : value(v) {
        std::cout << "Config 构造" << std::endl;
    }
    ~Config() { std::cout << "Config 析构" << std::endl; }
    int get() const { return value; }
};

// 工厂函数返回unique_ptr
std::unique_ptr<Config> createConfig(int value) {
    return std::make_unique<Config>(value);
}

// 接收所有权
void processConfig(std::unique_ptr<Config> config) {
    std::cout << "配置值: " << config->get() << std::endl;
}

int main() {
    auto cfg = createConfig(42);
    processConfig(std::move(cfg)); // 转移所有权
    // cfg现在为空
    return 0;
}
```

**输出：**
```
Config 构造
配置值: 42
Config 析构
```

## 三、注意事项与常见陷阱

- **不能拷贝，只能移动**：赋值或传值必须用`std::move`。
- **`make_unique`是C++14特性**：C++11需手写或用`new`。
- **`get()`返回裸指针不转移所有权**：不要用裸指针创建另一个`unique_ptr`。
- **`release()`后需手动`delete`**：返回的裸指针不会自动释放。
- **`sizeof(unique_ptr) == sizeof(ptr)`**：与裸指针大小相同，零额外开销。
