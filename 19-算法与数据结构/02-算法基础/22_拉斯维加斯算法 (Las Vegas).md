## Las Vegas


```javascript
拉斯维加斯算法总是给出正确结果，但运行时间是随机的。```


```
// 拉斯维加斯：随机化快速排序
function randomQsort(arr) {
  if (arr.length <= 1) return arr;
  const pivot = arr[Math.floor(Math.random() * arr.length)];
  const left = [], right = [], equal = [];
  for (const x of arr) {
    if (x < pivot) left.push(x);
    else if (x > pivot) right.push(x);
    else equal.push(x);
  }
  return [...randomQsort(left), ...equal, ...randomQsort(right)];
}
console.log(randomQsort([3,1,4,1,5,9,2,6])); // 总是正确排序```


  点击按钮查看结果
