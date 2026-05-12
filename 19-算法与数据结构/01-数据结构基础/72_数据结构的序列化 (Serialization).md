# 73-数据结构的序列化 (Serialization)

序列化将数据结构转换为字符串/字节流，反序列化是其逆过程。常用于数据传输和存储。

## 二叉树序列化

```javascript
// 前序遍历序列化
function serialize(root) {
  if (!root) return '#';
  return `${root.val},${serialize(root.left)},${serialize(root.right)}`;
}

function deserialize(s) {
  const arr = s.split(',');
  let idx = 0;

  function dfs() {
    if (arr[idx] === '#') { idx++; return null; }
    const node = { val: Number(arr[idx++]), left: null, right: null };
    node.left = dfs();
    node.right = dfs();
    return node;
  }
  return dfs();
}

// 层序遍历序列化
function serializeLevel(root) {
  if (!root) return '';
  const q = [root], result = [];
  while (q.length) {
    const node = q.shift();
    if (node) {
      result.push(node.val);
      q.push(node.left, node.right);
    } else {
      result.push('#');
    }
  }
  return result.join(',');
}

function deserializeLevel(s) {
  if (!s) return null;
  const arr = s.split(',');
  const root = { val: Number(arr[0]), left: null, right: null };
  const q = [root];
  let i = 1;
  while (q.length) {
    const node = q.shift();
    if (arr[i] !== '#') {
      node.left = { val: Number(arr[i]), left: null, right: null };
      q.push(node.left);
    }
    i++;
    if (arr[i] !== '#') {
      node.right = { val: Number(arr[i]), left: null, right: null };
      q.push(node.right);
    }
    i++;
  }
  return root;
}
```

## C++ 实现

```cpp
#include <string>
#include <sstream>
using namespace std;

struct TreeNode {
    int val;
    TreeNode *left, *right;
    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
};

// 前序序列化
string serialize(TreeNode* root) {
    if (!root) return "#";
    return to_string(root->val) + "," + serialize(root->left) + "," + serialize(root->right);
}

TreeNode* deserialize(istringstream& ss) {
    string token;
    if (!getline(ss, token, ',')) return nullptr;
    if (token == "#") return nullptr;
    TreeNode* node = new TreeNode(stoi(token));
    node->left = deserialize(ss);
    node->right = deserialize(ss);
    return node;
}
```

## 链表序列化

```javascript
function serializeList(head) {
  const result = [];
  while (head) {
    result.push(head.val);
    head = head.next;
  }
  return result.join(',');
}

function deserializeList(s) {
  if (!s) return null;
  const vals = s.split(',').map(Number);
  const dummy = { val: 0, next: null };
  let curr = dummy;
  for (const v of vals) {
    curr.next = { val: v, next: null };
    curr = curr.next;
  }
  return dummy.next;
}
```

## 图的序列化

```javascript
// 邻接表序列化
function serializeGraph(adjList) {
  return JSON.stringify(adjList);
}

function deserializeGraph(s) {
  return JSON.parse(s);
}

// BFS 序列化含环的图
function serializeGraphBFS(graph) {
  const visited = new Set();
  const result = [];
  for (const node of Object.keys(graph)) {
    if (!visited.has(node)) {
      const queue = [node];
      visited.add(node);
      while (queue.length) {
        const curr = queue.shift();
        result.push(`${curr}:${(graph[curr] || []).join('|')}`);
        for (const nei of graph[curr] || []) {
          if (!visited.has(nei)) {
            visited.add(nei);
            queue.push(nei);
          }
        }
      }
    }
  }
  return result.join(';');
}
```

## 复杂度分析

| 结构 | 序列化 | 反序列化 | 空间 |
|------|--------|---------|------|
| 二叉树 | O(n) | O(n) | O(n) |
| 链表 | O(n) | O(n) | O(n) |
| 图 | O(V+E) | O(V+E) | O(V+E) |

## 序列化格式对比

| 格式 | 可读性 | 大小 | 速度 | 兼容性 |
|------|--------|------|------|--------|
| JSON | 高 | 较大 | 中 | 广泛 |
| 二进制 | 低 | 小 | 快 | 特定系统 |
| Protobuf | 低 | 小 | 快 | 跨语言 |
| 自定义文本 | 中 | 中 | 中 | 特定场景 |

## 常见陷阱

1. **空节点处理**：二叉树序列化要表示空节点
2. **环检测**：图序列化要处理环，避免无限递归
3. **类型恢复**：反序列化时要恢复正确的数据类型
4. **特殊字符**：包含分隔符的字符串需要转义
