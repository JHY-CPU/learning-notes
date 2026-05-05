## Expression Evaluation


```javascript
利用两个栈分别存储操作数和操作符，可以实现中缀表达式求值。```


```
// 中缀表达式求值
function calc(expr) {
  const prec = {'+':1,'-':1,'*':2,'/':2};
  const nums=[], ops=[];
  const apply = () => {
    const b=nums.pop(), a=nums.pop(), op=ops.pop();
    nums.push(op==='+'?a+b:op==='-'?a-b:op==='*'?a*b:Math.floor(a/b));
  };
  for (const ch of expr) {
    if (ch >= '0' && ch <= '9') nums.push(Number(ch));
    else if (ch in prec) {
      while (ops.length && ops[ops.length-1] !== '(' && prec[ops[ops.length-1]] >= prec[ch]) apply();
      ops.push(ch);
    } else if (ch === '(') ops.push(ch);
    else if (ch === ')') { while (ops[ops.length-1] !== '(') apply(); ops.pop(); }
  }
  while (ops.length) apply();
  return nums[0];
}
console.log(calc("1+2*3")); // 7```


  点击按钮查看结果
