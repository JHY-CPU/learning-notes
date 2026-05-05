## Assign Cookies


```javascript
贪心地满足最多孩子：给每个孩子分配不小于其胃口的最小饼干。```


```
function findContentChildren(g, s) {
  g.sort((a,b) => a-b);
  s.sort((a,b) => a-b);
  let child = 0, cookie = 0;
  while (child < g.length && cookie < s.length) {
    if (s[cookie] >= g[child]) child++;
    cookie++;
  }
  return child;
}
console.log(findContentChildren([1,2,3], [1,1])); // 1
console.log(findContentChildren([1,2], [1,2,3])); // 2```


  点击按钮查看结果
