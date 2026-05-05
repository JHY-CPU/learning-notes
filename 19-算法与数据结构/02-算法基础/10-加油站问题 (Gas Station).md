## Gas Station


```javascript
在环形路线上找到能跑完全程的加油站起点。```


```
function canCompleteCircuit(gas, cost) {
  let total = 0, curr = 0, start = 0;
  for (let i = 0; i < gas.length; i++) {
    total += gas[i] - cost[i];
    curr += gas[i] - cost[i];
    if (curr < 0) { start = i + 1; curr = 0; }
  }
  return total >= 0 ? start : -1;
}
console.log(canCompleteCircuit([1,2,3,4,5], [3,4,5,1,2])); // 3
console.log(canCompleteCircuit([2,3,4], [3,4,3])); // -1```


  点击按钮查看结果
