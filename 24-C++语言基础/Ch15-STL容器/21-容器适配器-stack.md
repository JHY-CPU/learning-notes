# 容器适配器：stack

## 一、概念说明

`std::stack`是LIFO（后进先出）容器适配器（C++标准 §23.6.5.2），封装了底层容器的接口，只暴露栈操作（push、pop、top）。默认基于`deque`，也可基于`vector`或`list`。stack不是独立的容器，而是对已有容器的接口限制。

### 1.1 设计理念

```
容器适配器模式：
- 不直接管理内存
- 依赖底层容器（deque/vector/list）
- 只暴露特定接口（LIFO）
- 编译期多态（模板参数指定底层容器）
```

## 二、具体用法

### 2.1 基本操作

```cpp
#include <stack>
#include <iostream>
#include <vector>
#include <list>

int main() {
    // 默认基于deque
    std::stack<int> stk;

    // 基于vector
    std::stack<int, std::vector<int>> vec_stk;

    // 基于list
    std::stack<int, std::list<int>> list_stk;

    // 压栈
    stk.push(1);
    stk.push(2);
    stk.push(3);
    stk.emplace(4);  // C++11，原地构造

    // 访问栈顶
    std::cout << "top: " << stk.top() << std::endl;  // 4

    // 弹栈（无返回值！）
    stk.pop();  // 移除4
    std::cout << "top: " << stk.top() << std::endl;  // 3

    // 大小和空判断
    std::cout << "size: " << stk.size() << std::endl;  // 3
    std::cout << "empty: " << stk.empty() << std::endl;  // 0

    // 遍历（需要逐个弹出，会破坏栈）
    while (!stk.empty()) {
        std::cout << stk.top() << " ";
        stk.pop();
    }
    // 3 2 1
}
```

### 2.2 底层容器选择

```cpp
/*
| 底层容器 | push/pop性能 | 内存特性 | 适用场景 |
|---------|-------------|---------|---------|
| deque   | O(1)        | 分段连续 | 默认选择 |
| vector  | O(1)均摊    | 连续    | 缓存友好 |
| list    | O(1)        | 非连续  | 稳定引用 |
*/
```

### 2.3 实用示例

```cpp
#include <string>
#include <stack>

// 括号匹配
bool is_valid_parentheses(const std::string& s) {
    std::stack<char> stk;
    for (char c : s) {
        if (c == '(' || c == '[' || c == '{') {
            stk.push(c);
        } else {
            if (stk.empty()) return false;
            char top = stk.top();
            stk.pop();
            if ((c == ')' && top != '(') ||
                (c == ']' && top != '[') ||
                (c == '}' && top != '{'))
                return false;
        }
    }
    return stk.empty();
}

// 后缀表达式求值
#include <sstream>
double eval_rpn(const std::string& expr) {
    std::stack<double> stk;
    std::istringstream iss(expr);
    std::string token;
    while (iss >> token) {
        if (token == "+" || token == "-" || token == "*" || token == "/") {
            double b = stk.top(); stk.pop();
            double a = stk.top(); stk.pop();
            if (token == "+") stk.push(a + b);
            else if (token == "-") stk.push(a - b);
            else if (token == "*") stk.push(a * b);
            else stk.push(a / b);
        } else {
            stk.push(std::stod(token));
        }
    }
    return stk.top();
}
```

## 三、注意事项与常见陷阱

1. **`pop()`不返回值**：需要先`top()`获取再`pop()`，分开调用是为了异常安全
2. **不支持遍历**：只能访问栈顶，没有迭代器
3. **底层容器需支持`back`、`push_back`、`pop_back`**
4. **stack没有`clear`方法**：清空需要循环pop，或重新构造
5. **适用场景**：括号匹配、表达式求值、DFS、函数调用栈模拟
6. **`emplace`（C++11）**：直接在栈顶构造元素
