## Top K Problem


```javascript
从数组中找出第 K 大（小）的元素，可用堆或快速选择。```


```
// 堆方法求第K大
function findKthLargest(nums, k) {
  // 小顶堆，保持大小为k
  const minHeap = [];
  for (const n of nums) {
    minHeap.push(n);
    if (minHeap.length > k) {
      minHeap.sort((a,b) => a-b);
      minHeap.shift();
    }
  }
  minHeap.sort((a,b) => a-b);
  return minHeap[0];
}
// 快速选择
function quickSelect(nums, k) {
  const idx = nums.length - k;
  function select(l, r) {
    const p = partition(nums, l, r);
    if (p === idx) return nums[p];
    return p < idx ? select(p+1, r) : select(l, p-1);
  }
  return select(0, nums.length-1);
}
function partition(arr, l, r) {
  const pivot = arr[r]; let i = l;
  for (let j = l; j < r; j++)
    if (arr[j] <= pivot) { [arr[i], arr[j]] = [arr[j], arr[i]]; i++; }
  [arr[i], arr[r]] = [arr[r], arr[i]];
  return i;
}
console.log(findKthLargest([3,2,1,5,6,4], 2)); // 5
console.log(quickSelect([3,2,1,5,6,4], 2)); // 5```


  点击按钮查看结果
