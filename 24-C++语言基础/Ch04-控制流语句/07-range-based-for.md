# range-based for

## 一、概念说明

C++11引入的范围for循环简化了容器遍历语法。它自动处理迭代器的begin/end，使代码更简洁。

## 二、具体用法

### 2.1 基本用法

```cpp
#include <iostream>
#include <vector>
#include <string>
using namespace std;

int main() {
    // 遍历数组
    int arr[] = {1, 2, 3, 4, 5};
    for (int x : arr) {
        cout << x << " ";
    }
    cout << endl;

    // 遍历STL容器
    vector<string> names = {"Alice", "Bob", "Charlie"};
    for (const auto& name : names) {
        cout << name << " ";
    }
    cout << endl;

    // 遍历字符串
    string str = "Hello";
    for (char c : str) {
        cout << c << "-";
    }
    cout << endl;

    // 遍历初始化列表
    for (int x : {10, 20, 30, 40}) {
        cout << x << " ";
    }
    cout << endl;

    return 0;
}
```

输出：
```
1 2 3 4 5
Alice Bob Charlie
H-e-l-l-o-
10 20 30 40
```

### 2.2 引用与值

```cpp
#include <iostream>
#include <vector>
using namespace std;

int main() {
    vector<int> nums = {1, 2, 3, 4, 5};

    // 值：拷贝元素（不修改原容器）
    for (int x : nums) {
        x *= 2;  // 不影响nums
    }

    // 引用：可以修改原容器
    for (int& x : nums) {
        x *= 2;  // 修改nums
    }
    for (int x : nums) cout << x << " ";  // 2 4 6 8 10
    cout << endl;

    // const引用：只读，避免拷贝（推荐用于大对象）
    vector<string> words = {"hello", "world"};
    for (const auto& w : words) {
        cout << w << " ";
        // w = "modified";  // 编译错误：const
    }
    cout << endl;

    return 0;
}
```

输出：
```
2 4 6 8 10
hello world
```

### 2.3 自定义类型支持range-based for

```cpp
#include <iostream>
using namespace std;

// 自定义容器支持range-based for
class IntRange {
    int start, end_;
public:
    IntRange(int s, int e) : start(s), end_(e) {}

    // 需要begin()和end()
    class Iterator {
        int current;
    public:
        Iterator(int val) : current(val) {}
        int operator*() const { return current; }
        Iterator& operator++() { ++current; return *this; }
        bool operator!=(const Iterator& other) const {
            return current != other.current;
        }
    };

    Iterator begin() const { return Iterator(start); }
    Iterator end() const { return Iterator(end_); }
};

int main() {
    for (int i : IntRange(1, 6)) {
        cout << i << " ";
    }
    cout << endl;

    return 0;
}
```

输出：
```
1 2 3 4 5
```

## 三、注意事项与常见陷阱

1. **避免不必要的拷贝**：遍历大对象时使用`const auto&`
2. **修改元素用`auto&`**：不加引用无法修改原容器
3. **不要在range-for中增删元素**：可能导致迭代器失效
4. **临时对象**：`for (auto x : getVector())`中getVector返回的临时对象在循环期间有效
5. **不能获取索引**：需要索引时用传统for或`std::distance`
