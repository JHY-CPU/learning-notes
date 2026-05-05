## Greedy Basics


```javascript
贪心算法每一步选择当前最优解，局部最优导致全局最优。```


```
// 零钱兑换（贪心 - 适用规范货币）
function coinChange(coins, amount) {
  coins.sort((a,b) => b - a);
  let count = 0;
  for (const c of coins) {
    while (amount >= c) { amount -= c; count++; }
  }
  return amount === 0 ? count : -1;
}
console.log(coinChange([25,10,5,1], 63)); // 25+25+10+1+1+1 = 6个
console.log(coinChange([1,5,10,25], 63)); // 6个```


  点击按钮查看结果
