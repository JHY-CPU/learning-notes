## Stack & DFS


```javascript
栈可以模拟递归实现深度优先搜索，避免递归调用栈溢出。```


```
// 用栈实现 DFS
function dfs(graph, start) {
  const visited = new Set();
  const stack = [start];
  const result = [];
  while (stack.length) {
    const node = stack.pop();
    if (!visited.has(node)) {
      visited.add(node);
      result.push(node);
      for (const nei of graph[node] || []) {
        if (!visited.has(nei)) stack.push(nei);
      }
    }
  }
  return result;
}
const g = {A:['B','C'],B:['D'],C:['E'],D:[],E:[]};
console.log(dfs(g, 'A')); // ['A','C','E','B','D']```


  点击按钮查看结果
