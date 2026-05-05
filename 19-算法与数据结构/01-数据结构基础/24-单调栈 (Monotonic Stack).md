## Monotonic Stack


```javascript
单调栈中的元素保持单调递增或递减，常用于找下一个更大/更小元素。```


```
// 下一个更大元素
function nextGreater(arr) {
  const res = new Array(arr.length).fill(-1);
  const stack = [];
  for (let i = 0; i < arr.length; i++) {
    while (stack.length && arr[stack[stack.length-1]] < arr[i]) {
      res[stack.pop()] = arr[i];
    }
    stack.push(i);
  }
  return res;
}
console.log(nextGreater([2,1,3,4,2])); // [3,3,4,-1,-1]```


  点击按钮查看结果
