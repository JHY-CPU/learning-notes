## Generate Parentheses


```javascript
生成所有有效的括号组合，是回溯+剪枝的经典问题。```


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


  点击按钮查看结果
