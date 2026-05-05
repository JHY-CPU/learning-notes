## Serialization


```javascript
序列化将数据结构转换为字符串/字节流，反序列化是其逆过程。```


```
// 二叉树序列化（前序遍历）
function serialize(root) {
  if (!root) return '#';
  return `${root.val},${serialize(root.left)},${serialize(root.right)}`;
}
function deserialize(s) {
  const arr = s.split(',');
  let idx = 0;
  function dfs() {
    if (arr[idx] === '#') { idx++; return null; }
    const node = {val: Number(arr[idx]), left: null, right: null};
    idx++;
    node.left = dfs();
    node.right = dfs();
    return node;
  }
  return dfs();
}
const tree = {val:1,left:{val:2,left:null,right:null},right:{val:3,left:{val:4,left:null,right:null},right:null}};
const ser = serialize(tree);
console.log(ser); // 1,2,#,#,3,4,#,#,#
console.log(deserialize(ser)); // restored```


  点击按钮查看结果
