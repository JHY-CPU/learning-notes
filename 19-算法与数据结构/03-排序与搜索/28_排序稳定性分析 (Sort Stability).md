## Sort Stability


```javascript
稳定排序保持相等元素的相对顺序，不稳定排序则不保证。```


```
// 稳定排序：冒泡、插入、归并、计数、基数
// 不稳定排序：选择、快排、堆排、希尔

// 演示稳定性的重要性
const students = [
  {name:'Alice', grade:85},
  {name:'Bob', grade:90},
  {name:'Charlie', grade:85},
  {name:'David', grade:80}
];
// 稳定排序：先按grade排序后，相同grade保持原顺序
function stableSort(arr, key) {
  return arr.slice().sort((a,b) => {
    if (a[key] < b[key]) return -1;
    if (a[key] > b[key]) return 1;
    return 0;
  });
}
console.log(stableSort(students, 'grade'));
// 相同grade的学生保持输入时的顺序```


  点击按钮查看结果
