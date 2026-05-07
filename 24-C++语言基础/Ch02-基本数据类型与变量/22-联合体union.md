# 联合体 union

## 一、概念说明

联合体（union）是一种特殊的数据结构，所有成员**共享同一块内存**。同一时间只能存储一个成员的值。C++17引入了`std::variant`作为类型安全的替代方案。

## 二、具体用法

### 2.1 基本联合体

```cpp
#include <iostream>
using namespace std;

union Data {
    int i;
    float f;
    char c;
    char str[8];
};

int main() {
    Data d;

    // 同一时间只有一个成员有效
    d.i = 42;
    cout << "int: " << d.i << endl;
    cout << "float: " << d.f << " (垃圾值)" << endl;

    d.f = 3.14f;
    cout << "float: " << d.f << endl;
    cout << "int: " << d.i << " (垃圾值，因为f覆盖了i)" << endl;

    d.str[0] = 'H';
    d.str[1] = 'i';
    d.str[2] = '\0';
    cout << "str: " << d.str << endl;

    cout << "union大小: " << sizeof(Data) << " 字节" << endl;
    // 大小等于最大成员的大小（考虑对齐）

    return 0;
}
```

输出（示例）：
```
int: 42
float: 5.88545e-44 (垃圾值)
float: 3.14
int: 1078523331 (垃圾值，因为f覆盖了i)
str: Hi
union大小: 8 字节
```

### 2.2 匿名联合体

```cpp
#include <iostream>
#include <cstring>
using namespace std;

struct Variant {
    enum Type { Int, Float, String } type;

    union {             // 匿名联合体
        int intVal;
        float floatVal;
        char strVal[32];
    };

    void setInt(int v) {
        type = Int;
        intVal = v;
    }

    void setFloat(float v) {
        type = Float;
        floatVal = v;
    }

    void setString(const char* v) {
        type = String;
        strncpy(strVal, v, 31);
        strVal[31] = '\0';
    }

    void print() const {
        switch (type) {
            case Int:    cout << "int: " << intVal << endl; break;
            case Float:  cout << "float: " << floatVal << endl; break;
            case String: cout << "string: " << strVal << endl; break;
        }
    }
};

int main() {
    Variant v;
    v.setInt(42);
    v.print();

    v.setFloat(3.14f);
    v.print();

    v.setString("Hello");
    v.print();

    return 0;
}
```

输出：
```
int: 42
float: 3.14
string: Hello
```

### 2.3 std::variant替代方案（C++17推荐）

```cpp
#include <iostream>
#include <variant>
#include <string>
using namespace std;

int main() {
    // std::variant：类型安全的联合体
    variant<int, double, string> data;

    data = 42;
    cout << "int: " << get<int>(data) << endl;

    data = 3.14;
    cout << "double: " << get<double>(data) << endl;

    data = "Hello"s;
    cout << "string: " << get<string>(data) << endl;

    // 类型安全：错误的get会抛异常
    try {
        cout << get<int>(data) << endl;  // data当前是string
    } catch (const bad_variant_access& e) {
        cout << "异常: 类型不匹配" << endl;
    }

    // 检查当前类型
    cout << "当前类型index: " << data.index() << endl;
    cout << "是否是string: " << holds_alternative<string>(data) << endl;

    // visit：访问者模式
    visit([](auto&& val) {
        cout << "visit得到: " << val << endl;
    }, data);

    return 0;
}
```

输出：
```
int: 42
double: 3.14
string: Hello
异常: 类型不匹配
当前类型index: 2
是否是string: 1
visit得到: Hello
```

## 三、注意事项与常见陷阱

1. **同一时间只有一个成员有效**：写入一个成员后，其他成员的值是不确定的
2. **不能包含非平凡类型**：C++11前的union不能包含有构造函数/析构函数的类型
3. **匿名union只能在类/函数内定义**：不能在全局作用域使用匿名union
4. **优先使用std::variant**：类型安全、自动管理生命周期、支持访问者模式
5. **union的大小**：等于最大成员的大小（考虑对齐填充）
