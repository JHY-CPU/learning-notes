# 42-哈希表与分组 (Hash Table Grouping)

哈希表可以将具有相同特征的元素归为一组，用特征值作为键，同组元素列表作为值。

## 字母异位词分组

```javascript
// 排序法：排序后的字符串作为键
function groupAnagrams(strs) {
  const map = new Map();
  for (const s of strs) {
    const key = s.split('').sort().join('');
    if (!map.has(key)) map.set(key, []);
    map.get(key).push(s);
  }
  return [...map.values()];
}

// 计数法：避免排序，O(n*k) 优于 O(n*k*logk)
function groupAnagramsCount(strs) {
  const map = new Map();
  for (const s of strs) {
    const count = new Array(26).fill(0);
    for (const ch of s) count[ch.charCodeAt(0) - 97]++;
    const key = count.join('#');
    if (!map.has(key)) map.set(key, []);
    map.get(key).push(s);
  }
  return [...map.values()];
}

console.log(groupAnagrams(["eat","tea","tan","ate","nat","bat"]));
// [["eat","tea","ate"],["tan","nat"],["bat"]]
```

## C++ 实现

```cpp
#include <vector>
#include <string>
#include <unordered_map>
#include <algorithm>
using namespace std;

vector<vector<string>> groupAnagrams(vector<string>& strs) {
    unordered_map<string, vector<string>> map;
    for (const string& s : strs) {
        string key = s;
        sort(key.begin(), key.end());
        map[key].push_back(s);
    }
    vector<vector<string>> result;
    for (auto& [_, group] : map) result.push_back(group);
    return result;
}
```

## 同构字符串分组

```javascript
// 模式编码：将字符串映射到模式
// "egg" -> "011", "add" -> "011"（同构）
function groupIsomorphic(strs) {
  const map = new Map();
  for (const s of strs) {
    const pattern = encodePattern(s);
    if (!map.has(pattern)) map.set(pattern, []);
    map.get(pattern).push(s);
  }
  return [...map.values()];
}

function encodePattern(s) {
  const map = {};
  let code = 0;
  let result = '';
  for (const ch of s) {
    if (!(ch in map)) map[ch] = code++;
    result += map[ch] + ',';
  }
  return result;
}
```

## 按特征分组

```javascript
// 按字符串长度分组
function groupByLength(strs) {
  const map = new Map();
  for (const s of strs) {
    const len = s.length;
    if (!map.has(len)) map.set(len, []);
    map.get(len).push(s);
  }
  return [...map.values()];
}

// 按字符频率分组
function groupByFrequency(s) {
  const freq = {};
  for (const ch of s) freq[ch] = (freq[ch] || 0) + 1;
  const map = new Map();
  for (const [ch, count] of Object.entries(freq)) {
    if (!map.has(count)) map.set(count, []);
    map.get(count).push(ch);
  }
  return map;
}
```

## 复杂度分析

| 方法 | 时间 | 空间 |
|------|------|------|
| 异位词（排序） | O(n * k log k) | O(n * k) |
| 异位词（计数） | O(n * k) | O(n * k) |
| 同构分组 | O(n * k) | O(n * k) |
| 按特征分组 | O(n) | O(n) |

n = 字符串数量，k = 字符串平均长度。

## 其他分组应用

```javascript
// 丑数分组
function groupUglyNumbers(nums) {
  return nums.reduce((groups, n) => {
    const factors = getPrimeFactors(n);
    const key = factors.sort().join(',');
    (groups[key] = groups[key] || []).push(n);
    return groups;
  }, {});
}

// 链表分组（按值范围）
function groupByRange(head, rangeSize) {
  const map = new Map();
  let curr = head;
  while (curr) {
    const bucket = Math.floor(curr.val / rangeSize);
    if (!map.has(bucket)) map.set(bucket, []);
    map.get(bucket).push(curr.val);
    curr = curr.next;
  }
  return map;
}
```

## 何时使用哈希表分组

- 需要将元素按某种特征归类
- 特征可以被高效地计算为哈希键
- 需要 O(n) 时间完成分组
- 替代排序后线性扫描的方法

## 常见陷阱

1. **特征键设计**：不同组的元素不能映射到同一键
2. **对象键**：JavaScript Map 中对象键比较引用
3. **空字符串**：需要特殊处理空字符串
4. **溢出**：计数法中拼接数字要加分隔符避免歧义
