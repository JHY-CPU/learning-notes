## Algorithm Design Patterns


```javascript
常见算法设计模式：暴力、贪心、分治、动态规划、回溯、分支限界。```


```
// 算法模式对比
const patterns = [
  { name: '暴力', approach: '枚举所有可能', use: '小规模数据', ex: '冒泡排序' },
  { name: '贪心', approach: '局部最优选择', use: '最优子结构', ex: 'Dijkstra' },
  { name: '分治', approach: '分解→解决→合并', use: '可分解问题', ex: '归并排序' },
  { name: 'DP', approach: '子问题+状态转移', use: '重叠子问题', ex: '背包问题' },
  { name: '回溯', approach: '试探+回退', use: '约束满足', ex: 'N皇后' },
];
console.table(patterns);```


  点击按钮查看结果
