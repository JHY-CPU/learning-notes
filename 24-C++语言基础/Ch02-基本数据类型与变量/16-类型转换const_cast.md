# 类型转换 const_cast

## 一、概念说明

`const_cast`用于**添加或移除**变量的`const`或`volatile`属性。它是最危险的类型转换之一，因为移除const并修改原本是const的对象会导致**未定义行为**。

## 二、具体用法

### 2.1 移除const属性

```cpp
#include <iostream>
using namespace std;

void legacyFunction(int* ptr) {
    // 假设这是一个不接受const的旧C库函数
    *ptr = 100;
}

int main() {
    const int x = 42;
    const int* cp = &x;

    // 移除const限定
    int* p = const_cast<int*>(cp);

    // 警告：修改const对象是未定义行为！
    // *p = 100;  // UB：x是const的

    // 安全用法：原始对象本身不是const
    int y = 42;
    const int* cpy = &y;      // 指向非const对象的const指针
    int* py = const_cast<int*>(cpy);
    *py = 100;                // OK：y本身不是const
    cout << "y = " << y << endl;

    return 0;
}
```

输出：
```
y = 100
```

### 2.2 合法使用场景

```cpp
#include <iostream>
#include <string>
using namespace std;

// 场景1：调用不接受const的旧API
void oldApiPrint(char* str) {
    cout << "旧API: " << str << endl;
}

void newApiWrapper(const string& message) {
    // message是const，但oldApiPrint需要非const
    // 如果确定oldApiPrint不会修改内容，可以用const_cast
    oldApiPrint(const_cast<char*>(message.c_str()));
}

// 场景2：const成员函数中修改mutable不够用的情况
class Cache {
    mutable int cacheValue = 0;
    bool cached = false;

public:
    int compute() const {
        if (!cached) {
            // 使用const_cast修改"逻辑上非const"的成员
            auto self = const_cast<Cache*>(this);
            self->cacheValue = expensiveCalculation();
            self->cached = true;
        }
        return cacheValue;
    }

private:
    int expensiveCalculation() const {
        return 42;  // 假装很耗时的计算
    }
};

int main() {
    // 使用包装函数
    string msg = "Hello";
    newApiWrapper(msg);

    // 使用缓存
    Cache c;
    cout << "计算结果: " << c.compute() << endl;
    cout << "再次计算: " << c.compute() << endl;  // 使用缓存

    return 0;
}
```

输出：
```
旧API: Hello
计算结果: 42
再次计算: 42
```

### 2.3 const_cast的限制

```cpp
#include <iostream>
using namespace std;

int main() {
    // const_cast只能用于指针和引用
    // int x = const_cast<int>(42);  // 错误：不是指针或引用

    // const_cast不能改变类型本身
    double d = 3.14;
    // int* ip = const_cast<int*>(&d);  // 错误：类型不同

    // 只能去除const/volatile
    const int* cp = new int(42);
    int* p = const_cast<int*>(cp);  // OK：去除const

    // 实际应用：避免代码重复
    auto process = [](const int* arr, int size) {
        // 需要对数组排序，但不想修改const版本的逻辑
        int* mutableArr = const_cast<int*>(arr);
        // 如果调用者传入真正的const数组，这里就是UB
        for (int i = 0; i < size - 1; i++) {
            if (mutableArr[i] > mutableArr[i + 1]) {
                swap(mutableArr[i], mutableArr[i + 1]);
            }
        }
    };

    int nums[]{5, 3, 1, 4, 2};
    process(nums, 5);
    for (int n : nums) cout << n << " ";
    cout << endl;

    delete cp;
    return 0;
}
```

输出：
```
3 1 4 2 5
```

## 三、注意事项与常见陷阱

1. **修改真正的const对象是UB**：如果原始对象声明为const，移除const后修改它是未定义行为
2. **不要在日常代码中使用const_cast**：它通常是设计问题的信号，应重构代码
3. **const_cast只能修改const/volatile**：不能用它转换不相关的类型
4. **编译器可能优化**：编译器可能假设const对象不会改变，导致使用const_cast修改后的值不被更新
5. **优先考虑mutable**：如果是const成员函数需要修改成员，优先使用`mutable`而非`const_cast`
