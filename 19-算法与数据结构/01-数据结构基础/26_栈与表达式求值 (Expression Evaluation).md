# 27-栈与表达式求值 (Expression Evaluation)

利用两个栈分别存储操作数和操作符，可以实现中缀表达式求值。栈是表达式解析的核心工具。

## 中缀表达式求值（双栈法）

```javascript
// 使用两个栈：操作数栈和操作符栈
function calculate(expr) {
  const prec = { '+': 1, '-': 1, '*': 2, '/': 2 };
  const nums = [], ops = [];

  const apply = () => {
    const b = nums.pop(), a = nums.pop(), op = ops.pop();
    if (op === '+') nums.push(a + b);
    else if (op === '-') nums.push(a - b);
    else if (op === '*') nums.push(a * b);
    else nums.push(Math.trunc(a / b));
  };

  for (const ch of expr) {
    if (ch === ' ') continue;
    if (ch >= '0' && ch <= '9') {
      nums.push(Number(ch));
    } else if (ch === '(') {
      ops.push(ch);
    } else if (ch === ')') {
      while (ops[ops.length - 1] !== '(') apply();
      ops.pop(); // 弹出 '('
    } else if (ch in prec) {
      while (ops.length && ops[ops.length - 1] !== '(' &&
             prec[ops[ops.length - 1]] >= prec[ch]) {
        apply();
      }
      ops.push(ch);
    }
  }
  while (ops.length) apply();
  return nums[0];
}

console.log(calculate("1+2*3"));      // 7
console.log(calculate("(1+2)*3"));    // 9
console.log(calculate("2*(3+4)-5"));  // 9
```

## C++ 实现

```cpp
#include <string>
#include <stack>
#include <unordered_map>
using namespace std;

int calculate(string expr) {
    unordered_map<char, int> prec = {{'+',1},{'-',1},{'*',2},{'/',2}};
    stack<int> nums;
    stack<char> ops;

    auto apply = [&]() {
        int b = nums.top(); nums.pop();
        int a = nums.top(); nums.pop();
        char op = ops.top(); ops.pop();
        if (op == '+') nums.push(a + b);
        else if (op == '-') nums.push(a - b);
        else if (op == '*') nums.push(a * b);
        else nums.push(a / b);
    };

    for (int i = 0; i < expr.size(); i++) {
        char ch = expr[i];
        if (ch == ' ') continue;
        if (isdigit(ch)) {
            nums.push(ch - '0');
        } else if (ch == '(') {
            ops.push(ch);
        } else if (ch == ')') {
            while (ops.top() != '(') apply();
            ops.pop();
        } else {
            while (!ops.empty() && ops.top() != '(' &&
                   prec[ops.top()] >= prec[ch]) apply();
            ops.push(ch);
        }
    }
    while (!ops.empty()) apply();
    return nums.top();
}
```

## 后缀表达式（逆波兰）求值

```javascript
// 后缀表达式求值：只需要一个操作数栈
function evalRPN(tokens) {
  const stack = [];
  for (const t of tokens) {
    if (['+', '-', '*', '/'].includes(t)) {
      const b = stack.pop(), a = stack.pop();
      if (t === '+') stack.push(a + b);
      else if (t === '-') stack.push(a - b);
      else if (t === '*') stack.push(a * b);
      else stack.push(Math.trunc(a / b));
    } else {
      stack.push(Number(t));
    }
  }
  return stack[0];
}

console.log(evalRPN(["2", "1", "+", "3", "*"])); // 9
```

## 中缀转后缀（调度场算法）

```javascript
function infixToPostfix(expr) {
  const prec = { '+': 1, '-': 1, '*': 2, '/': 2 };
  const ops = [], output = [];

  for (const ch of expr) {
    if (ch === ' ') continue;
    if (ch >= '0' && ch <= '9') {
      output.push(ch);
    } else if (ch === '(') {
      ops.push(ch);
    } else if (ch === ')') {
      while (ops[ops.length - 1] !== '(') output.push(ops.pop());
      ops.pop();
    } else {
      while (ops.length && ops[ops.length - 1] !== '(' &&
             prec[ops[ops.length - 1]] >= prec[ch]) {
        output.push(ops.pop());
      }
      ops.push(ch);
    }
  }
  while (ops.length) output.push(ops.pop());
  return output;
}

console.log(infixToPostfix("1+2*3")); // ['1','2','3','*','+']
```

## 基本计算器（含多位数和空格）

```javascript
function calculateAdvanced(s) {
  let stack = [], num = 0, sign = '+';
  for (let i = 0; i <= s.length; i++) {
    const ch = s[i];
    if (ch >= '0' && ch <= '9') {
      num = num * 10 + Number(ch);
    } else if (ch === '+' || ch === '-' || ch === '*' || ch === '/' || i === s.length) {
      if (sign === '+') stack.push(num);
      else if (sign === '-') stack.push(-num);
      else if (sign === '*') stack.push(stack.pop() * num);
      else stack.push(Math.trunc(stack.pop() / num));
      sign = ch;
      num = 0;
    }
  }
  return stack.reduce((a, b) => a + b, 0);
}
```

## 复杂度分析

| 方法 | 时间 | 空间 |
|------|------|------|
| 双栈中缀求值 | O(n) | O(n) |
| 后缀求值 | O(n) | O(n) |
| 中缀转后缀 | O(n) | O(n) |

## 常见陷阱

1. **除法方向**：`a / b` 中 a 先出栈，b 后出栈
2. **负数处理**：中缀表达式中的负号需要特殊处理
3. **多位数字**：不能只处理单字符，需要拼接多位数
4. **空格跳过**：忽略表达式中的空格字符
5. **栈空检查**：操作前确保栈中有足够元素
