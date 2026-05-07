# Rule of Five

## 一、概念说明

**Rule of Five**（五法则是C++11对Rule of Three的扩展：**如果类需要自定义析构函数、拷贝构造、拷贝赋值、移动构造、移动赋值中的任何一个，通常需要定义全部五个**。C++11引入了移动语义后，Rule of Three扩展为Rule of Five。

### 1.1 特殊成员函数总结

| 特殊成员函数 | 未定义时的行为 |
|-------------|--------------|
| 析构函数 | 编译器合成 |
| 拷贝构造 | 编译器合成（成员逐一拷贝） |
| 拷贝赋值 | 编译器合成（成员逐一赋值） |
| 移动构造 | 若无拷贝构造则合成 |
| 移动赋值 | 若无拷贝赋值则合成 |

## 二、具体用法

### 2.1 完整的Rule of Five实现

```cpp
#include <iostream>
#include <algorithm>
#include <utility>

class DynamicArray {
private:
    int* data;
    size_t size;
public:
    // 构造函数
    explicit DynamicArray(size_t n) : data(new int[n]()), size(n) {}

    // ① 析构函数
    ~DynamicArray() {
        delete[] data;
        std::cout << "析构" << std::endl;
    }

    // ② 拷贝构造
    DynamicArray(const DynamicArray& o) : data(new int[o.size]), size(o.size) {
        std::copy(o.data, o.data + size, data);
        std::cout << "拷贝构造" << std::endl;
    }

    // ③ 拷贝赋值
    DynamicArray& operator=(const DynamicArray& o) {
        if (this != &o) {
            delete[] data;
            size = o.size;
            data = new int[size];
            std::copy(o.data, o.data + size, data);
        }
        std::cout << "拷贝赋值" << std::endl;
        return *this;
    }

    // ④ 移动构造
    DynamicArray(DynamicArray&& o) noexcept
        : data(o.data), size(o.size) {
        o.data = nullptr;
        o.size = 0;
        std::cout << "移动构造" << std::endl;
    }

    // ⑤ 移动赋值
    DynamicArray& operator=(DynamicArray&& o) noexcept {
        if (this != &o) {
            delete[] data;
            data = o.data;
            size = o.size;
            o.data = nullptr;
            o.size = 0;
        }
        std::cout << "移动赋值" << std::endl;
        return *this;
    }

    size_t getSize() const { return size; }
};

int main() {
    DynamicArray a(10);
    DynamicArray b = a;          // 拷贝构造
    DynamicArray c = std::move(a); // 移动构造
    DynamicArray d(5);
    d = b;                       // 拷贝赋值
    d = std::move(c);            // 移动赋值
    std::cout << "d.size=" << d.getSize() << std::endl;
    return 0;
}
```

**输出：**
```
拷贝构造
移动构造
拷贝赋值
移动赋值
d.size=10
析构
析构
析构
析构
```

## 三、注意事项与常见陷阱

- Rule of Five的优先级低于Rule of Zero
- 定义了析构函数会**阻止编译器生成移动操作**（退化为拷贝）
- 定义了任何拷贝操作也会**阻止移动操作的自动生成**
- 移动操作必须标记`noexcept`
- 现代C++应优先使用智能指针和容器，避免手动实现Rule of Five
