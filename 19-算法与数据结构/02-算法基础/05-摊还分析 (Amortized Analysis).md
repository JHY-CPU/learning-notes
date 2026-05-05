## Amortized Analysis


```javascript
摊还分析将偶尔的高成本操作均摊到所有操作中，得到平均成本。```


```
// 动态数组的摊还分析
class DynamicArray {
  constructor() { this.data = new Array(1); this.size = 0; this.capacity = 1; }
  push(x) {
    if (this.size === this.capacity) {
      const newData = new Array(this.capacity * 2);
      for (let i = 0; i < this.size; i++) newData[i] = this.data[i];
      this.data = newData;
      this.capacity *= 2;
    }
    this.data[this.size++] = x;
  }
}
// 每次 push 的摊还成本 = O(1)
// n次push的总成本 = O(n), 平均 O(1)
console.log('动态数组的摊还分析：扩容时O(n)，均摊后O(1)');```


  点击按钮查看结果
