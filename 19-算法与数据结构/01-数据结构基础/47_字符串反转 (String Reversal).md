## String Reversal


```javascript
字符串反转是基础操作，有多种实现方式：双指针、递归、栈。```


```
function reverseStr(s) {
  return s.split('').reverse().join('');
}
function reverseStr2(s) {
  let r = '';
  for (let i = s.length-1; i >= 0; i--) r += s[i];
  return r;
}
function reverseWords(s) {
  return s.split(' ').filter(w => w).reverse().join(' ');
}
console.log(reverseStr("algorithm")); // mhtirogla
console.log(reverseWords("hello world")); // world hello```


  点击按钮查看结果
