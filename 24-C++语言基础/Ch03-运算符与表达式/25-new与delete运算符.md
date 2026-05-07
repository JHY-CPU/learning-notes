# new与delete运算符

## 一、概念说明

`new`在**堆**上分配内存并调用构造函数，`delete`调用析构函数并释放内存。C++的`new`/`delete`比C的`malloc`/`free`更安全，但现代C++推荐使用智能指针。

## 二、具体用法

### 2.1 基本new/delete

```cpp
#include <iostream>
using namespace std;

int main() {
    // new分配单个对象
    int* p = new int(42);
    cout << "*p = " << *p << endl;
    delete p;

    // new[]分配数组
    int* arr = new int[5]{1, 2, 3, 4, 5};
    for (int i = 0; i < 5; i++) {
        cout << arr[i] << " ";
    }
    cout << endl;
    delete[] arr;  // 必须用delete[]释放数组

    // new分配对象
    struct Point {
        double x, y;
        Point(double x, double y) : x(x), y(y) {
            cout << "Point构造" << endl;
        }
        ~Point() { cout << "Point析构" << endl; }
    };

    Point* pt = new Point(3.0, 4.0);
    cout << "pt = (" << pt->x << ", " << pt->y << ")" << endl;
    delete pt;

    return 0;
}
```

输出：
```
*p = 42
1 2 3 4 5
Point构造
pt = (3, 4)
Point析构
```

### 2.2 new失败处理

```cpp
#include <iostream>
#include <new>
using namespace std;

int main() {
    // 默认：失败时抛出std::bad_alloc
    try {
        // 申请巨大的内存
        size_t huge = 1ULL << 50;  // 1PB
        int* p = new int[huge];
        delete[] p;
    } catch (const bad_alloc& e) {
        cout << "new失败（异常）: " << e.what() << endl;
    }

    // nothrow：失败时返回nullptr
    int* p = new(nothrow) int[1000000];
    if (p) {
        cout << "分配成功" << endl;
        delete[] p;
    } else {
        cout << "new失败（返回null）" << endl;
    }

    return 0;
}
```

输出：
```
new失败（异常）: std::bad_alloc
分配成功
```

### 2.3 placement new

```cpp
#include <iostream>
#include <new>
using namespace std;

struct Message {
    int id;
    char text[32];

    Message(int i, const char* t) : id(i) {
        strncpy(text, t, 31);
        text[31] = '\0';
        cout << "Message构造: " << id << endl;
    }

    ~Message() {
        cout << "Message析构: " << id << endl;
    }
};

int main() {
    // 在预分配的内存上构造对象
    alignas(Message) char buffer[sizeof(Message)];

    // placement new：在指定地址构造
    Message* msg = new (buffer) Message(1, "Hello");

    cout << "消息: " << msg->id << ", " << msg->text << endl;

    // 必须手动调用析构函数
    msg->~Message();

    // buffer本身的内存不需要释放（栈上分配）
    return 0;
}
```

输出：
```
Message构造: 1
消息: 1, Hello
Message析构: 1
```

## 三、注意事项与常见陷阱

1. **new/delete必须配对**：`new`配`delete`，`new[]`配`delete[]`
2. **delete[]不能用于单对象**：反之亦然，混用是未定义行为
3. **delete后指针不变**：`delete p`后p仍指向原地址（悬垂指针），应置为`nullptr`
4. **智能指针替代**：优先使用`make_unique`和`make_shared`而非裸`new`
5. **内存泄漏**：异常发生时裸new分配的内存可能泄漏，智能指针自动管理
