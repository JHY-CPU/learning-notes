# 容器适配器：stack

## 一、概念说明

`std::stack`是LIFO（后进先出）容器适配器，封装了底层容器的接口，只暴露栈操作。默认基于`deque`。

## 二、具体用法

```cpp
#include <stack>
#include <iostream>
#include <vector>

int main() {
    // 默认基于deque
    std::stack<int> stk;

    // 基于vector
    std::stack<int, std::vector<int>> vec_stk;

    // 操作
    stk.push(1);
    stk.push(2);
    stk.push(3);

    std::cout << "top: " << stk.top() << std::endl;  // 3
    std::cout << "size: " << stk.size() << std::endl; // 3

    stk.pop();  // 移除3（无返回值）
    std::cout << "top: " << stk.top() << std::endl;  // 2

    // 遍历（需要逐个弹出）
    while (!stk.empty()) {
        std::cout << stk.top() << " ";
        stk.pop();
    }
    // 2 1
}
```

## 三、注意事项

- `pop()`不返回值，需要先`top()`获取
- 不支持遍历（只能访问栈顶）
- 底层容器需要支持`back`、`push_back`、`pop_back`
- 应用：括号匹配、表达式求值、DFS
