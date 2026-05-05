## Correctness Proof


```javascript
证明算法正确性的常用方法：循环不变式、数学归纳法、反证法。```


```
// 循环不变式证明
// 插入排序的循环不变式：
// 每次迭代开始时，子数组 arr[0..i-1] 已经有序
function insertionSort(arr) {
  for (let i = 1; i < arr.length; i++) {
    const key = arr[i];
    let j = i - 1;
    while (j >= 0 && arr[j] > key) { arr[j+1] = arr[j]; j--; }
    arr[j+1] = key;
  }
  return arr;
}
// 贪心算法的交换论证
// 分治算法代入法
console.log(insertionSort([5,2,4,6,1,3])); // [1,2,3,4,5,6]```


  点击按钮查看结果
