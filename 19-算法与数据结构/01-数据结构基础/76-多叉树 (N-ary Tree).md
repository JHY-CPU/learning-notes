## N-ary Tree


```javascript
多叉树的每个节点可以有多个子节点，常用于文件系统、组织架构等。```


```
class NAryNode {
  constructor(val) { this.val = val; this.children = []; }
}
// 多叉树前序遍历
function preorder(root) {
  if (!root) return [];
  const res = [root.val];
  for (const child of root.children) res.push(...preorder(child));
  return res;
}
// 多叉树层序遍历
function levelOrder(root) {
  if (!root) return [];
  const q = [root], res = [];
  while (q.length) {
    const len = q.length;
    const level = [];
    for (let i = 0; i < len; i++) {
      const n = q.shift();
      level.push(n.val);
      q.push(...n.children);
    }
    res.push(level);
  }
  return res;
}```


  点击按钮查看结果
