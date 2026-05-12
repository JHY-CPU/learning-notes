# Structure Selection

### 如何选择合适的数据结构

选择数据结构需要根据操作模式（读多写少 vs 写多读少）、数据规模、有序性需求和内存约束综合考虑。

### 关键特性

- **读多写少**：哈希表或数组，查找效率优先
- **写多读少**：链表或跳表，插入删除效率优先
- **需要有序**：平衡 BST、跳表或有序数组
- **需要最值**：堆（优先队列）
- **需要去重**：哈希集合或平衡 BST 集合

### 选择决策树

1. 需要键值映射？
   - 无序且频繁查找 -> 哈希表
   - 需要有序遍历 -> 平衡 BST / TreeMap
2. 只需存储元素？
   - 需要去重 -> Set（哈希或树）
   - 允许重复 -> 数组或链表
3. 需要频繁取最值？ -> 堆
4. 需要按层遍历？ -> 队列
5. 需要回退操作？ -> 栈

### 适用场景 vs 替代方案

- **通用场景**：哈希表是最安全的默认选择
- **数据库索引**：B+ 树（范围查询）或哈希索引（点查）
- **实时系统**：数组或环形缓冲区（可预测性能）
- **内存受限**：紧凑结构如位图、布隆过滤器

### 常见陷阱

- 过早优化：先用最简单的结构，性能有问题再换
- 忽略常数因子：理论 O(log n) 可能因常数大而慢于 O(n)
- 忘记考虑内存对齐和缓存行的影响

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
console.log(recommendDS({frequentOp:'min/max', dataSize:10000}));
```


### 实际应用

- **Redis**：根据数据类型选择不同底层结构（ziplist、skiplist、hashtable）
- **数据库**：根据查询模式选择 B+ 树或哈希索引
- **前端框架**：React 用哈希表管理组件状态，Vue 用响应式数组
- **游戏引擎**：根据查询频率选择空间分区结构（四叉树、网格）

  点击按钮查看结果
