## Space Complexity


```javascript
数据结构的空间复杂度包括存储数据本身和额外辅助空间。```


```
// 空间复杂度对比
// 数组: O(n) - 连续存储
// 链表: O(n) - 每个节点额外存储指针
// 哈希表: O(n) - 存储键值对+桶
// 二叉树: O(n) - 每个节点存储左右指针
// 图(邻接矩阵): O(V²)
// 图(邻接表): O(V+E)
// 并查集: O(n)
// 布隆过滤器: O(k) - 固定空间，与n无关
function estimateMemory(n, type) {
  const sizes = { array: n * 8, linkedList: n * 24, hashTable: n * 40, tree: n * 32, graphMatrix: n * n * 8, graphList: n * 24 };
  return sizes[type] || '未知';
}
console.log(`100万元素数组约 ${estimateMemory(1000000, 'array') / 1024 / 1024} MB`);```


  点击按钮查看结果
