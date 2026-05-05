## 基数排序 (Radix Sort)

  基数排序是一种非比较排序，按位对数字进行排序。可以按最低有效位（LSD）或最高有效位（MSD）进行排序。


>
    **复杂度分析：**时间复杂度 O(d*(n+k))，其中 d 是位数，k 是基数。空间 O(n+k)。稳定排序。


  ## LSD 基数排序（从低位到高位）


    - 从最低有效位（个位）开始，对所有元素按该位进行稳定排序

    - 依次处理十位、百位……直到最高位

    - 每轮使用计数排序作为稳定排序子程序



  ## 代码实现


```
function radixSort(arr) {
  const max = Math.max(...arr);
  const maxDigits = String(max).length;

  for (let pos = 0; pos < maxDigits; pos++) {
    const buckets = Array.from({length: 10}, () => []);
    for (const num of arr) {
      const digit = Math.floor(num / Math.pow(10, pos)) % 10;
      buckets[digit].push(num);
    }
    arr = [].concat(...buckets);
  }
  return arr;
}```

  ## 交互演示
