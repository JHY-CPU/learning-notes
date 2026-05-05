## 15-差分数组 (Difference Array)

差分数组是前缀和的逆运算，专门用于高效处理数组的区间批量增减操作。

## 差分数组原理

```javascript

// 原始数组 arr
// 差分数组 diff: diff[i] = arr[i] - arr[i-1] (i > 0)
// diff[0] = arr[0]

// 核心性质：对原数组区间 [l, r] 加 val
// 只需 diff[l] += val 和 diff[r + 1] -= val
// 最后对差分数组求前缀和还原原数组

class Difference {
  constructor(nums) {
    this.diff = new Array(nums.length);
    this.diff[0] = nums[0];
    for (let i = 1; i < nums.length; i++) {
      this.diff[i] = nums[i] - nums[i - 1];
    }
  }

  // 对区间 [l, r] 增加 val
  increment(l, r, val) {
    this.diff[l] += val;
    if (r + 1 < this.diff.length) {
      this.diff[r + 1] -= val;
    }
  }

  // 从差分数组还原结果
  getResult() {
    let res = new Array(this.diff.length);
    res[0] = this.diff[0];
    for (let i = 1; i < this.diff.length; i++) {
      res[i] = res[i - 1] + this.diff[i];
    }
    return res;
  }
}
```

## 应用场景

```javascript

// 经典问题：航班预订统计
// bookings = [[1,2,10], [2,3,20], [2,5,25]]
// n = 5 个航班
// 每个 [l, r, val] 表示给 l 到 r 航班各预订 val 个座位
function corpFlightBookings(bookings, n) {
  let diff = new Array(n + 1).fill(0);

  for (let [l, r, val] of bookings) {
    diff[l] += val;
    if (r + 1 <= n) diff[r + 1] -= val;
  }

  // 还原
  let result = new Array(n);
  result[0] = diff[1];
  for (let i = 1; i < n; i++) {
    result[i] = result[i - 1] + diff[i + 1];
  }
  return result;
}
```

## 交互演示
