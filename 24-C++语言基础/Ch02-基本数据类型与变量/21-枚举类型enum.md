# 枚举类型 enum

## 一、概念说明

枚举类型定义了一组**命名的整数常量**，使代码更可读、更安全。C++11引入了有作用域的`enum class`，解决了传统枚举的作用域污染和隐式转换问题。

## 二、具体用法

### 2.1 传统枚举

```cpp
#include <iostream>
using namespace std;

// 无作用域枚举（C风格）
enum Color {
    Red,       // 0
    Green,     // 1
    Blue,      // 2
    Yellow = 10,
    Orange     // 11
};

// 指定底层类型
enum Priority : char {
    Low = 'L',
    Medium = 'M',
    High = 'H'
};

int main() {
    Color c = Blue;
    cout << "Blue = " << c << endl;
    cout << "Orange = " << Orange << endl;

    // 问题：枚举值泄漏到外层作用域
    // int Red = 42;  // 编译错误：Red已被定义

    // 问题：隐式转换为int
    int val = c;       // OK但不安全
    cout << "Blue -> int: " << val << endl;

    // 问题：不同枚举可以比较
    Priority p = High;
    // if (c == p) {}  // 可以比较（虽然无意义）

    return 0;
}
```

输出：
```
Blue = 2
Orange = 11
Blue -> int: 2
```

### 2.2 有作用域枚举 enum class（C++11推荐）

```cpp
#include <iostream>
using namespace std;

// enum class：有作用域，不泄漏，不隐式转换
enum class Direction {
    Up,       // Direction::Up
    Down,     // Direction::Down
    Left,     // Direction::Left
    Right     // Direction::Right
};

// 指定底层类型
enum class HttpStatus : int {
    OK = 200,
    NotFound = 404,
    ServerError = 500
};

// 另一个枚举，不会与Direction冲突
enum class Color {
    Red,      // Color::Red（与Direction的值不同名无冲突）
    Green,
    Blue
};

int main() {
    Direction dir = Direction::Up;
    Color color = Color::Red;

    // 必须用完整名称
    // Direction d = Up;  // 编译错误

    // 不隐式转换为int
    // int x = dir;  // 编译错误
    int x = static_cast<int>(dir);  // 必须显式转换

    // 不同枚举不能比较
    // if (dir == color) {}  // 编译错误

    // 可以用于switch
    switch (dir) {
        case Direction::Up:    cout << "向上" << endl; break;
        case Direction::Down:  cout << "向下" << endl; break;
        case Direction::Left:  cout << "向左" << endl; break;
        case Direction::Right: cout << "向右" << endl; break;
    }

    cout << "HttpStatus::OK = " << static_cast<int>(HttpStatus::OK) << endl;

    return 0;
}
```

输出：
```
向上
HttpStatus::OK = 200
```

### 2.3 枚举的常用操作

```cpp
#include <iostream>
#include <vector>
using namespace std;

enum class Permission : unsigned int {
    None    = 0,
    Read    = 1 << 0,   // 1
    Write   = 1 << 1,   // 2
    Execute = 1 << 2,   // 4
    All     = Read | Write | Execute  // 7
};

// 位运算重载
Permission operator|(Permission a, Permission b) {
    return static_cast<Permission>(
        static_cast<unsigned int>(a) | static_cast<unsigned int>(b));
}

bool hasPermission(Permission user, Permission required) {
    return (static_cast<unsigned int>(user) &
            static_cast<unsigned int>(required)) != 0;
}

int main() {
    Permission userPerm = Permission::Read | Permission::Write;

    cout << "有读权限: " << (hasPermission(userPerm, Permission::Read) ? "是" : "否") << endl;
    cout << "有执行权限: " << (hasPermission(userPerm, Permission::Execute) ? "是" : "否") << endl;

    return 0;
}
```

输出：
```
有读权限: 是
有执行权限: 否
```

## 三、注意事项与常见陷阱

1. **优先使用enum class**：除非需要与C代码兼容，否则不用传统enum
2. **传统enum会污染作用域**：`enum Color { Red };`后`Red`在外部作用域可见
3. **隐式转int的问题**：传统enum会隐式转为int，可能导致逻辑错误
4. **底层类型**：默认enum的底层类型由编译器决定，enum class可以指定
5. **switch要覆盖所有值**：编译器可以检查switch是否覆盖了所有枚举值
