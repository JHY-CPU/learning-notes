## String Encoding


```javascript
字符串编码处理特殊字符，如 URL 编码、Base64、游程编码。```


```
// 游程编码
function runLengthEncode(s) {
  let res = '', count = 1;
  for (let i = 0; i < s.length; i++) {
    if (s[i] === s[i+1]) count++;
    else { res += count + s[i]; count = 1; }
  }
  return res;
}
function runLengthDecode(s) {
  let res = '';
  for (let i = 0; i < s.length; i+=2) {
    res += s[i+1].repeat(Number(s[i]));
  }
  return res;
}
console.log(runLengthEncode("AAABBBCCC")); // 3A3B3C```


  点击按钮查看结果
