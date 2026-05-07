# C++11 enum class

## 一、概念说明

`enum class`（有作用域枚举）解决传统`enum`的问题：名称泄漏到外层作用域、隐式转换为整数。枚举值必须通过`枚举类::值`访问。

```cpp
#include <iostream>

// 传统enum（问题多）
enum Color { Red, Green, Blue };
// enum Fruit { Apple, Orange }; // 错误：Red, Green等名称冲突

// enum class（推荐）
enum class ColorClass { Red, Green, Blue };
enum class FruitClass { Apple, Orange, Banana };

int main() {
    ColorClass c = ColorClass::Red;
    FruitClass f = FruitClass::Apple;

    // 不能隐式转换为int
    // int x = c; // 错误
    int x = static_cast<int>(c);
    std::cout << "Red = " << x << std::endl;

    // 可以比较
    if (c == ColorClass::Red) {
        std::cout << "是红色" << std::endl;
    }

    return 0;
}
```

**输出：**
```
Red = 0
是红色
```

## 二、具体用法

### 2.1 指定底层类型

```cpp
#include <iostream>

// 指定底层类型
enum class Status : uint8_t {
    Ok = 0,
    Error = 1,
    Pending = 2
};

enum class Direction : int {
    North = 0,
    South = 180,
    East = 90,
    West = 270
};

std::string toString(Status s) {
    switch (s) {
        case Status::Ok: return "OK";
        case Status::Error: return "ERROR";
        case Status::Pending: return "PENDING";
        default: return "UNKNOWN";
    }
}

int main() {
    Status s = Status::Ok;
    std::cout << "Status: " << toString(s) << std::endl;
    std::cout << "sizeof(Status): " << sizeof(Status) << " 字节" << std::endl;

    Direction d = Direction::North;
    std::cout << "North值: " << static_cast<int>(d) << std::endl;

    return 0;
}
```

**输出：**
```
Status: OK
sizeof(Status): 1 字节
North值: 0
```

## 三、注意事项与常见陷阱

- **`enum class`不会隐式转换为整数**：需要`static_cast`。
- **`enum class`不会泄漏名称到外层作用域**：避免命名冲突。
- **可以指定底层类型**：默认`int`，可以是任何整数类型。
- **`enum class`不支持位运算（需要自定义）**：或使用`std::underlying_type`。
- **switch语句建议覆盖所有枚举值**：编译器可以检查。
