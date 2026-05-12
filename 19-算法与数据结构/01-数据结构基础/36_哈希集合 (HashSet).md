# 37-哈希集合 (HashSet)

哈希集合基于哈希表实现，只存储键不存储值，提供 O(1) 的添加、删除和存在性判断。

## JavaScript 实现

```javascript
// JavaScript 内置 Set
const set = new Set();
set.add(1); set.add(2); set.add(3);
console.log(set.has(2)); // true
console.log(set.size);   // 3
set.delete(2);
console.log(set.has(2)); // false

// 集合运算
const a = new Set([1, 2, 3]);
const b = new Set([2, 3, 4]);
const union = new Set([...a, ...b]);           // {1,2,3,4}
const intersection = new Set([...a].filter(x => b.has(x))); // {2,3}
const difference = new Set([...a].filter(x => !b.has(x))); // {1}
```

## 手动实现 HashSet

```javascript
class HashSet {
  constructor(capacity = 16) {
    this.buckets = new Array(capacity).fill(null).map(() => []);
    this.size = 0;
    this.capacity = capacity;
    this.loadFactor = 0.75;
  }

  _hash(key) {
    let hash = 0;
    const str = String(key);
    for (let i = 0; i < str.length; i++) {
      hash = (hash * 31 + str.charCodeAt(i)) % this.capacity;
    }
    return hash;
  }

  add(key) {
    const idx = this._hash(key);
    const bucket = this.buckets[idx];
    for (const k of bucket) {
      if (k === key) return false; // 已存在
    }
    bucket.push(key);
    this.size++;
    if (this.size > this.capacity * this.loadFactor) this._resize();
    return true;
  }

  has(key) {
    const idx = this._hash(key);
    return this.buckets[idx].includes(key);
  }

  delete(key) {
    const idx = this._hash(key);
    const bucket = this.buckets[idx];
    const i = bucket.indexOf(key);
    if (i === -1) return false;
    bucket.splice(i, 1);
    this.size--;
    return true;
  }

  _resize() {
    const old = this.buckets;
    this.capacity *= 2;
    this.buckets = new Array(this.capacity).fill(null).map(() => []);
    this.size = 0;
    for (const bucket of old) {
      for (const key of bucket) this.add(key);
    }
  }
}
```

## C++ 实现

```cpp
#include <unordered_set>
#include <iostream>
using namespace std;

int main() {
    unordered_set<int> s;
    s.insert(1); s.insert(2); s.insert(3);
    cout << s.count(2) << endl;  // 1 (存在)
    cout << s.count(5) << endl;  // 0 (不存在)
    s.erase(2);

    // 遍历
    for (int x : s) cout << x << " ";

    // 集合运算
    unordered_set<int> a = {1, 2, 3};
    unordered_set<int> b = {2, 3, 4};

    // 交集
    unordered_set<int> inter;
    for (int x : a) if (b.count(x)) inter.insert(x);
}
```

## 时间复杂度

| 操作 | 平均 | 最坏 |
|------|------|------|
| add | O(1) | O(n) |
| has | O(1) | O(n) |
| delete | O(1) | O(n) |
| 空间 | O(n) | O(n) |

## 典型应用

```javascript
// 数组去重
function removeDuplicates(nums) {
  return [...new Set(nums)];
}

// 存在重复元素
function containsDuplicate(nums) {
  return new Set(nums).size !== nums.length;
}

// 两个数组的交集
function intersection(nums1, nums2) {
  const set1 = new Set(nums1);
  return [...new Set(nums2)].filter(x => set1.has(x));
}

// 快乐数
function isHappy(n) {
  const seen = new Set();
  while (n !== 1 && !seen.has(n)) {
    seen.add(n);
    n = String(n).split('').reduce((s, d) => s + d * d, 0);
  }
  return n === 1;
}
```

## HashSet vs 数组

| 特性 | HashSet | 数组 |
|------|---------|------|
| 查找 | O(1) | O(n) |
| 去重 | 自动 | 需手动 |
| 内存 | 更多开销 | 更紧凑 |
| 有序性 | 无序 | 有序 |
| 范围查询 | 不支持 | 排序后支持 |

## 常见陷阱

1. **对象比较**：Set 中对象比较的是引用，不是值
2. **类型区分**：`"1"` 和 `1` 是不同元素
3. **遍历修改**：遍历 Set 时添加/删除元素可能导致问题
4. **JSON 序列化**：Set 不能直接 JSON.stringify
