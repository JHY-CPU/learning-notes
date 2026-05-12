# 76-多叉树 (N-ary Tree)

多叉树的每个节点可以有多个子节点，常用于文件系统、组织架构、DOM 树等场景。

## 节点定义

```javascript
class NAryNode {
  constructor(val) {
    this.val = val;
    this.children = [];
  }
}
```

## C++ 定义

```cpp
struct NAryNode {
    int val;
    vector<NAryNode*> children;
    NAryNode(int v) : val(v) {}
};
```

## 遍历方式

```javascript
// 前序遍历
function preorder(root) {
  if (!root) return [];
  const res = [root.val];
  for (const child of root.children) {
    res.push(...preorder(child));
  }
  return res;
}

// 后序遍历
function postorder(root) {
  if (!root) return [];
  const res = [];
  for (const child of root.children) {
    res.push(...postorder(child));
  }
  res.push(root.val);
  return res;
}

// 层序遍历
function levelOrder(root) {
  if (!root) return [];
  const queue = [root], res = [];
  while (queue.length) {
    const levelSize = queue.length;
    const level = [];
    for (let i = 0; i < levelSize; i++) {
      const node = queue.shift();
      level.push(node.val);
      queue.push(...node.children);
    }
    res.push(level);
  }
  return res;
}

// 迭代前序遍历
function preorderIterative(root) {
  if (!root) return [];
  const stack = [root], res = [];
  while (stack.length) {
    const node = stack.pop();
    res.push(node.val);
    for (let i = node.children.length - 1; i >= 0; i--) {
      stack.push(node.children[i]);
    }
  }
  return res;
}
```

## 最大深度

```javascript
function maxDepth(root) {
  if (!root) return 0;
  let maxChild = 0;
  for (const child of root.children) {
    maxChild = Math.max(maxChild, maxDepth(child));
  }
  return 1 + maxChild;
}

// BFS 版本
function maxDepthBFS(root) {
  if (!root) return 0;
  const queue = [root];
  let depth = 0;
  while (queue.length) {
    depth++;
    const size = queue.length;
    for (let i = 0; i < size; i++) {
      const node = queue.shift();
      queue.push(...node.children);
    }
  }
  return depth;
}
```

## 序列化

```javascript
function serializeNAry(root) {
  if (!root) return '';
  const result = [];
  function dfs(node) {
    result.push(String(node.val));
    result.push(String(node.children.length));
    for (const child of node.children) dfs(child);
  }
  dfs(root);
  return result.join(',');
}
```

## 多叉树 vs 二叉树

| 特性 | 多叉树 | 二叉树 |
|------|--------|--------|
| 子节点数 | 任意 | 最多 2 |
| 遍历方式 | 前/后/层序 | 前/中/后/层序 |
| 存储方式 | children 数组 | left/right 指针 |
| 转换 | 左孩子右兄弟法 | - |

## 左孩子右兄弟表示法

```javascript
// 多叉树转二叉树
// 每个节点的第一个子节点作为左孩子，其余兄弟作为右孩子
function toBinaryTree(root) {
  if (!root) return null;
  const bNode = { val: root.val, left: null, right: null };
  if (root.children.length > 0) {
    bNode.left = toBinaryTree(root.children[0]);
    let curr = bNode.left;
    for (let i = 1; i < root.children.length; i++) {
      curr.right = toBinaryTree(root.children[i]);
      curr = curr.right;
    }
  }
  return bNode;
}
```

## 应用场景

- **文件系统**：目录树结构
- **DOM 树**：HTML 文档结构
- **组织架构**：公司层级
- **决策树**：机器学习
- **JSON/XML 解析**：嵌套结构
- **游戏场景树**：游戏对象层级

## 常见陷阱

1. **空 children 数组**：叶子节点的 children 是空数组而非 null
2. **层序遍历**：用 queue 而非 stack
3. **深度计算**：空树深度为 0，单节点深度为 1
4. **内存占用**：大量子节点时 children 数组可能很大
