# 运行时类型信息（RTTI）

## 一、概念说明

**RTTI**（Run-Time Type Information）允许在运行时查询对象的实际类型。主要机制是`typeid`运算符和`type_info`类。RTTI需要至少一个虚函数才能工作。

## 二、具体用法

### 2.1 typeid运算符

```cpp
#include <iostream>
#include <typeinfo>

class Base {
public:
    virtual ~Base() = default;
};
class Derived : public Base {};

int main() {
    Base* b = new Derived();

    // typeid返回type_info引用
    std::cout << "b的类型: " << typeid(*b).name() << std::endl;
    std::cout << "Base类型: " << typeid(Base).name() << std::endl;

    // 比较实际类型
    if (typeid(*b) == typeid(Derived))
        std::cout << "b实际是Derived" << std::endl;

    delete b;
    return 0;
}
```

**输出：**
```
b的类型: 7Derived  （具体格式因编译器而异）
Base类型: 4Base
b实际是Derived
```

### 2.2 type_info成员

```cpp
#include <iostream>
#include <typeinfo>

int main() {
    const std::type_info& ti = typeid(int);
    std::cout << "name: " << ti.name() << std::endl;
    std::cout << "hash_code: " << ti.hash_code() << std::endl;
    return 0;
}
```

**输出：**
```
name: i  （因编译器而异）
hash_code: （实际数值）
```

## 三、注意事项与常见陷阱

- 没有虚函数的类，`typeid`只看静态类型
- `type_info::name()`返回的是实现定义的名称（可能被mangled）
- RTTI有运行时开销，某些嵌入式系统禁用RTTI
- 优先使用虚函数多态，只在必要时用`typeid`
