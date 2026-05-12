# C++20 span

## 一、概念说明

`std::span`（C++20 §24.7，`<span>`）是连续对象序列的非拥有视图。它结合了指针+长度的安全性，替代传统的`T*`+`size_t`参数对。类似`string_view`但适用于任意类型。

### 1.1 span vs 指针+长度

| 特性 | `T* + size_t` | `span<T>` |
|------|--------------|-----------|
| 边界检查 | 无 | 可选（`at()`） |
| 支持范围for | 否 | 是 |
| 子视图 | 手动计算 | `subspan`, `first`, `last` |
| 接受容器 | 需要`.data()` | 自动转换 |
| 编译期大小 | 不可知 | 可选（`span<T, N>`） |

```cpp
#include <iostream>
#include <span>
#include <vector>
#include <array>

// 用span代替指针+长度
void print(std::span<const int> data) {
    std::cout << "大小=" << data.size() << ": ";
    for (int x : data) std::cout << x << " ";
    std::cout << std::endl;
}

int main() {
    // 从不同来源创建span
    int arr[] = {1, 2, 3, 4, 5};
    std::vector<int> vec = {6, 7, 8};
    std::array<int, 3> stdarr = {9, 10, 11};

    print(arr);       // 从数组
    print(vec);       // 从vector
    print(stdarr);    // 从array

    // 动态span
    std::span<int> dynamicSpan(arr);
    dynamicSpan[0] = 100;
    std::cout << "修改后arr[0]=" << arr[0] << std::endl;

    // 固定大小span
    std::span<int, 5> fixedSpan(arr);
    std::cout << "固定大小: " << fixedSpan.size() << std::endl;

    return 0;
}
```

**输出：**
```
大小=5: 1 2 3 4 5
大小=3: 6 7 8
大小=3: 9 10 11
修改后arr[0]=100
固定大小: 5
```

## 二、具体用法

### 2.1 span的操作

```cpp
#include <iostream>
#include <span>

int main() {
    int data[] = {10, 20, 30, 40, 50};
    std::span<int> sp(data);

    // 子视图（零拷贝）
    auto sub = sp.subspan(1, 3); // [20, 30, 40]
    std::cout << "subspan(1,3): ";
    for (int x : sub) std::cout << x << " ";
    std::cout << std::endl;

    // first/last
    auto first2 = sp.first(2);
    auto last2 = sp.last(2);
    std::cout << "first(2): " << first2[0] << " " << first2[1] << std::endl;
    std::cout << "last(2): " << last2[0] << " " << last2[1] << std::endl;

    // data()获取裸指针
    int* ptr = sp.data();
    std::cout << "data(): " << *ptr << std::endl;

    // 元素访问
    std::cout << "front: " << sp.front() << std::endl;
    std::cout << "back: " << sp.back() << std::endl;
    std::cout << "size: " << sp.size() << std::endl;
    std::cout << "empty: " << sp.empty() << std::endl;

    return 0;
}
```

**输出：**
```
subspan(1,3): 20 30 40
first(2): 10 20
last(2): 40 50
data(): 10
front: 10
back: 50
size: 5
empty: 0
```

### 2.2 作为函数参数

```cpp
#include <iostream>
#include <span>
#include <vector>
#include <numeric>

// 通用函数：接受任何连续容器
int sum(std::span<const int> data) {
    return std::accumulate(data.begin(), data.end(), 0);
}

// 可写span
void doubleAll(std::span<int> data) {
    for (auto& x : data) x *= 2;
}

// 部分操作
void processFirst(std::span<int> data, size_t n) {
    auto first = data.first(n);
    for (auto& x : first) x += 10;
}

int main() {
    int arr[] = {1, 2, 3, 4, 5};
    std::vector<int> vec = {10, 20, 30};

    // 同一个函数接受不同容器
    std::cout << "sum(arr)=" << sum(arr) << std::endl;
    std::cout << "sum(vec)=" << sum(vec) << std::endl;

    // 修改
    doubleAll(arr);
    std::cout << "doubleAll后: ";
    for (int x : arr) std::cout << x << " ";
    std::cout << std::endl;

    return 0;
}
```

**输出：**
```
sum(arr)=15
sum(vec)=60
doubleAll后: 2 4 6 8 10
```

### 2.3 span与字节

```cpp
#include <iostream>
#include <span>
#include <cstring>

int main() {
    int x = 0x12345678;

    // as_bytes: 查看对象的字节表示
    auto bytes = std::as_bytes(std::span{&x, 1});
    std::cout << "x的字节: ";
    for (auto b : bytes) {
        std::cout << std::hex << static_cast<int>(b) << " ";
    }
    std::cout << std::endl;

    // as_writable_bytes: 可写字节视图
    auto writable = std::as_writable_bytes(std::span{&x, 1});
    writable[0] = 0xFF;

    std::cout << "修改后x: 0x" << std::hex << x << std::endl;

    return 0;
}
```

## 三、注意事项与常见陷阱

1. **`span`不拥有数据**：底层数据销毁后span悬垂。
2. **`span<T>`可写，`span<const T>`只读**：用`const span`限制修改。
3. **固定大小`span<T, N>`编译期检查大小**：更安全，但不够灵活。
4. **从`vector`创建span是O(1)**：不拷贝数据。
5. **`span`替代`T* + size_t`参数对**：更安全、更简洁。
6. **`span`可以隐式从`std::vector`和数组创建**：需要`<span>`头文件。
7. **详细内容参见Ch15 span视图章节**。
