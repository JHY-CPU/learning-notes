## Stack Applications


```javascript
栈在表达式求值、括号匹配、函数调用、撤销操作等场景广泛应用。```


```
// 括号匹配
function isBalanced(s) {
  const stack = [];
  const map = {')': '(', '}': '{', ']': '['};
  for (const ch of s) {
    if ('({['.includes(ch)) stack.push(ch);
    else if (map[ch] !== stack.pop()) return false;
  }
  return stack.length === 0;
}
console.log(isBalanced("(){}[]")); // true
console.log(isBalanced("({)}"));   // false```


  点击按钮查看结果
