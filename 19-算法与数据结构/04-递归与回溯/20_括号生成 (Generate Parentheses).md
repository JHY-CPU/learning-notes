# Generate Parentheses


```javascript
生成所有有效的括号组合，是回溯+剪枝的经典问题。```

## 概念说明

给定 n 对括号，生成所有格式正确的括号组合。有效的括号要求：任意前缀中左括号数 >= 右括号数，且最终左右括号数各为 n。这是一个回溯+剪枝的典型问题。

## 核心思路

用 `open` 和 `close` 分别记录已用的左括号和右括号数。递归过程中：若 `open < n` 可以加左括号；若 `close < open` 可以加右括号。这两条规则自然地保证了括号序列的合法性，无需额外验证。最终长度达到 2n 时收集结果。

## 复杂度分析

- **时间复杂度：** O(4^n / sqrt(n))，卡特兰数 C(n) 的增长量级。
- **空间复杂度：** O(n)，递归栈深度为 2n。

## 适用场景

- 括号匹配、表达式求值
- 符号串生成
- 卡特兰数相关应用（如二叉树计数、出栈序列）

```
function generateParenthesis(n) {
  const res = [];
  function backtrack(path, open, close) {
    if (path.length === 2 * n) { res.push(path); return; }
    if (open < n) backtrack(path + '(', open + 1, close);
    if (close < open) backtrack(path + ')', open, close + 1);
  }
  backtrack('', 0, 0);
  return res;
}
console.log(generateParenthesis(3)); // ["((()))","(()())","(())()","()(())","()()()"]```


## 常见变体与技巧

- **多种括号：** 扩展为同时包含 ()、[]、{} 时，需要检查括号类型匹配。
- **卡特兰数关系：** n 对括号的合法组合数 = 第 n 个卡特兰数 = C(2n,n)/(n+1)。
- **迭代法：** 用 BFS 逐层构造，每层加入一个 ( 或 )，同样按 open/close 规则剪枝。

## BFS 实现

```javascript
// BFS 逐层构造
function generateParenthesisBFS(n) {
  const queue = [['', 0, 0]];  // [str, open, close]
  const res = [];
  while (queue.length) {
    const [str, open, close] = queue.shift();
    if (str.length === 2 * n) { res.push(str); continue; }
    if (open < n) queue.push([str + '(', open + 1, close]);
    if (close < open) queue.push([str + ')', open, close + 1]);
  }
  return res;
}
```

## 卡特兰数

n 对括号的合法组合数 = 第 n 个卡特兰数：

```
C(0) = 1
C(n) = C(0)*C(n-1) + C(1)*C(n-2) + ... + C(n-1)*C(0)
     = C(2n, n) / (n + 1)

n=1: 1    ()
n=2: 2    ()(), (())
n=3: 5    ()()(), ()(()), (())(), (()()), ((()))
n=4: 14
n=5: 42
```

## LeetCode 相关题目

- 22. 括号生成
- 20. 有效的括号
- 32. 最长有效括号
- 301. 删除无效的括号

  点击按钮查看结果
