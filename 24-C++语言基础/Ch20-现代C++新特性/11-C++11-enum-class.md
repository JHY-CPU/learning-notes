# C++11 enum class

## 一、概念说明

`enum class`（有作用域枚举，C++11 §7.2）解决传统`enum`的两个核心问题：**名称泄漏**到外层作用域和**隐式转换**为整数。枚举值必须通过`枚举类::值`访问，不能隐式转为整数。

### 1.1 传统enum的问题

```cpp
// 传统enum的问题
enum Color { Red, Green, Blue };
enum Fruit { Apple, Orange, Banana }; // 编译错误！Red/Green/Blue名称泄漏

int x = Red; // 隐式转换为int，不安全
if (x == 1) { } // 含义不明确
```

```cpp
#include <iostream>

// 传统enum（问题多）
enum Color { Red, Green, Blue };

// enum class（推荐）
enum class ColorClass { Red, Green, Blue };
enum class FruitClass { Apple, Orange, Banana };

int main() {
    ColorClass c = ColorClass::Red;
    FruitClass f = FruitClass::Apple;

    // 不能隐式转换为int
    // int x = c; // 编译错误
    int x = static_cast<int>(c);
    std::cout << "Red = " << x << std::endl;

    // 同名枚举值不冲突
    // auto r1 = ColorClass::Red;
    // auto r2 = FruitClass::Apple; // 不同类型

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

可以为`enum class`指定底层整数类型，控制内存占用。

```cpp
#include <iostream>
#include <cstdint>

// 指定底层类型
enum class Status : uint8_t {
    Ok = 0,
    Error = 1,
    Pending = 2,
    Timeout = 3
};

enum class Direction : int {
    North = 0,
    South = 180,
    East = 90,
    West = 270
};

// 使用enum class的switch
std::string toString(Status s) {
    switch (s) {
        case Status::Ok: return "OK";
        case Status::Error: return "ERROR";
        case Status::Pending: return "PENDING";
        case Status::Timeout: return "TIMEOUT";
    }
    return "UNKNOWN"; // 防止警告
}

// 方向转角度
int toDegrees(Direction d) {
    return static_cast<int>(d);
}

int main() {
    Status s = Status::Ok;
    std::cout << "Status: " << toString(s) << std::endl;
    std::cout << "sizeof(Status): " << sizeof(Status) << " 字节" << std::endl;

    Direction d = Direction::North;
    std::cout << "North角度: " << toDegrees(d) << std::endl;

    Direction d2 = Direction::East;
    std::cout << "East角度: " << toDegrees(d2) << std::endl;

    return 0;
}
```

**输出：**
```
Status: OK
sizeof(Status): 1 字节
North角度: 0
East角度: 90
```

### 2.2 位运算支持

`enum class`不直接支持位运算，需要自定义运算符。

```cpp
#include <iostream>

enum class Permission : unsigned int {
    None    = 0,
    Read    = 1 << 0,  // 1
    Write   = 1 << 1,  // 2
    Execute = 1 << 2,  // 4
};

// 重载位运算
constexpr Permission operator|(Permission a, Permission b) {
    return static_cast<Permission>(
        static_cast<unsigned int>(a) | static_cast<unsigned int>(b));
}

constexpr Permission operator&(Permission a, Permission b) {
    return static_cast<Permission>(
        static_cast<unsigned int>(a) & static_cast<unsigned int>(b));
}

constexpr bool hasPermission(Permission set, Permission flag) {
    return (set & flag) == flag;
}

int main() {
    auto perms = Permission::Read | Permission::Write;

    std::cout << "有读权限: " << hasPermission(perms, Permission::Read) << std::endl;
    std::cout << "有执行权限: " << hasPermission(perms, Permission::Execute) << std::endl;

    return 0;
}
```

**输出：**
```
有读权限: 1
有执行权限: 0
```

### 2.3 遍历枚举值

```cpp
#include <iostream>

enum class Color { Red, Green, Blue, COUNT };

int main() {
    // 遍历枚举值
    for (int i = 0; i < static_cast<int>(Color::COUNT); ++i) {
        Color c = static_cast<Color>(i);
        std::cout << "Color " << i << std::endl;
    }

    return 0;
}
```

## 三、注意事项与常见陷阱

1. **`enum class`不会隐式转换为整数**：需要`static_cast`，这是设计目的——提高类型安全。
2. **`enum class`不会泄漏名称到外层作用域**：避免命名冲突，多个enum class可以有同名值。
3. **可以指定底层类型**：默认`int`，可以是任何整数类型（包括`uint8_t`节省内存）。
4. **`enum class`不支持位运算**：需要自定义重载或使用`std::underlying_type`。
5. **switch语句建议覆盖所有枚举值**：编译器可以检查完整性。
6. **C++17起可以用`using enum`引入枚举值**：减少重复的类名前缀。
