## Structure Selection


```javascript
不同场景应选择不同数据结构，选择依据包括操作模式、数据规模、内存限制。```


```
// 数据结构选择助手
function recommendDS(ops) {
  const { frequentOp, dataSize, needOrder, needUnique } = ops;
  if (frequentOp === 'search' && needOrder) return '平衡二叉搜索树';
  if (frequentOp === 'search' && !needOrder) return '哈希表';
  if (frequentOp === 'insert/delete' && needOrder) return '跳表';
  if (frequentOp === 'min/max') return '堆';
  if (dataSize > 1000000) return '应考虑外部存储或布隆过滤器';
  return '数组或链表即可';
}
console.log(recommendDS({frequentOp:'search', needOrder:true}));
console.log(recommendDS({frequentOp:'min/max', dataSize:10000}));```


  点击按钮查看结果
