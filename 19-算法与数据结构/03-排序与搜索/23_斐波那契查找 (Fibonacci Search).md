## 斐波那契查找 (Fibonacci Search)

  斐波那契查找基于斐波那契数列进行分割，使用黄金比例（约 0.618）决定查找位置，是二分查找的一种变体。


>
    **与二分查找的区别：**斐波那契查找只用加/减法运算确定分割点（二分查找需要除法/移位），在有些平台上可能更快。


  ## 斐波那契数列


```javascript
F(0) = 0
F(1) = 1
F(n) = F(n-1) + F(n-2)

前几项：0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144...```

  ## 算法步骤


    - 找到大于等于数组长度的最小斐波那契数 F(k)

    - 将数组长度扩展到 F(k) - 1（补最后一个元素）

    - 设置两个斐波那契指针：fibK = F(k), fibK1 = F(k-1), fibK2 = F(k-2)

    - 检查位置 index = min(offset + fibK2, n-1)：


      - 如果 target = arr[index]，返回 index

      - 如果 target > arr[index]，在右侧查找（调整 fibK1, fibK2）

      - 如果 target < arr[index]，在左侧查找




  ## 代码实现


```
function fibonacciSearch(arr, target) {
  const n = arr.length;
  let fibK2 = 0, fibK1 = 1, fibK = fibK2 + fibK1;

  while (fibK < n) {
    fibK2 = fibK1;
    fibK1 = fibK;
    fibK = fibK2 + fibK1;
  }

  let offset = -1;
  while (fibK > 1) {
    const i = Math.min(offset + fibK2, n - 1);
    if (arr[i] < target) {
      fibK = fibK1;
      fibK1 = fibK2;
      fibK2 = fibK - fibK1;
      offset = i;
    } else if (arr[i] > target) {
      fibK = fibK2;
      fibK1 = fibK1 - fibK2;
      fibK2 = fibK - fibK1;
    } else {
      return i;
    }
  }
  if (fibK1 === 1 && arr[offset + 1] === target) return offset + 1;
  return -1;
}```

  ## 交互演示
