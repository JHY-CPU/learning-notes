## Queue & BFS


```javascript
队列是实现广度优先搜索的核心数据结构，按层遍历图或树。```


```
// BFS 遍历树
function bfs(root) {
  if (!root) return [];
  const q = [root], res = [];
  while (q.length) {
    const node = q.shift();
    res.push(node.val);
    if (node.left) q.push(node.left);
    if (node.right) q.push(node.right);
  }
  return res;
}
const tree = {val:1, left:{val:2,left:null,right:null}, right:{val:3,left:{val:4,left:null,right:null},right:null}};
console.log(bfs(tree)); // [1,2,3,4]```


  点击按钮查看结果
