## Bitwise Basics


```javascript
位运算直接操作二进制位，包括与(&)、或(|)、异或(^)、取反(~)、左移(<<)、右移(>>)。```


```
const a = 0b1100, b = 0b1010;
console.log(`AND: ${a & b}`);  // 1000 = 8
console.log(`OR: ${a | b}`);   // 1110 = 14
console.log(`XOR: ${a ^ b}`);  // 0110 = 6
console.log(`NOT: ${~a}`);     // ...0011
console.log(`左移: ${a << 1}`); // 11000 = 24
console.log(`右移: ${a >> 1}`); // 110 = 6```


  点击按钮查看结果
