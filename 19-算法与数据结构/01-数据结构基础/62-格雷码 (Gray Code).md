## Gray Code


```javascript
格雷码是二进制编码的一种，相邻数字仅有一位不同，常用于数字信号传输。```


```
function grayCode(n) {
  const res = [];
  for (let i = 0; i < (1 << n); i++) {
    res.push(i ^ (i >> 1));
  }
  return res;
}
console.log(grayCode(3));
// [0,1,3,2,6,7,5,4]
// 二进制: [000,001,011,010,110,111,101,100]```


  点击按钮查看结果
