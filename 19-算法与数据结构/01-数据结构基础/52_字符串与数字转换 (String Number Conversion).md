## String Number Conversion


```javascript
atoi（字符串转整数）和 itoa（整数转字符串）是字符串处理的基本操作。```


```
function atoi(s) {
  s = s.trim();
  if (!s) return 0;
  let i = 0, sign = 1;
  if (s[i] === '+' || s[i] === '-') { if (s[i] === '-') sign = -1; i++; }
  let num = 0;
  while (i < s.length && s[i] >= '0' && s[i] <= '9') {
    num = num * 10 + (s[i].charCodeAt(0) - 48);
    if (num > 2147483647) return sign === 1 ? 2147483647 : -2147483648;
    i++;
  }
  return num * sign;
}
console.log(atoi("   -42")); // -42
console.log(atoi("4193 with words")); // 4193```


  点击按钮查看结果
