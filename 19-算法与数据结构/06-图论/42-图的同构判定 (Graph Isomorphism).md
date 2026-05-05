## Graph Isomorphism


```javascript
判断两个图是否同构是计算复杂性理论中的重要问题。```


```
// 图同构的简单判定（不一定完备）
function graphIsomorphismSimple(g1, g2, n) {
  // 必要条件：度序列相同
  const deg1 = new Array(n).fill(0).map((_,i) => (g1[i]||[]).length).sort((a,b)=>a-b);
  const deg2 = new Array(n).fill(0).map((_,i) => (g2[i]||[]).length).sort((a,b)=>a-b);
  for (let i = 0; i < n; i++) if (deg1[i] !== deg2[i]) return false;
  // 还需要进一步验证（Weisfeiler-Lehman 等）
  return '度序列相同，可能同构';
}
console.log(graphIsomorphismSimple({0:[1,2],1:[0],2:[0]}, {0:[1],1:[0,2],2:[1]}, 3));```


  点击按钮查看结果
