## Bitwise Addition


```javascript
用异或和与运算实现加法：异或得无进位和，与运算左移得进位。```


```
function add(a, b) {
  while (b !== 0) {
    const carry = a & b;
    a = a ^ b;
    b = carry << 1;
  }
  return a;
}
function subtract(a, b) { return add(a, add(~b, 1)); }
console.log(add(15, 27)); // 42
console.log(subtract(42, 15)); // 27
console.log(add(-5, 3)); // -2```


  点击按钮查看结果
