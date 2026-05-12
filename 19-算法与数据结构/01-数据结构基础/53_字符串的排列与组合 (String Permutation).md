# 54-字符串的排列与组合 (String Permutation)

字符串的全排列和组合是递归回溯的经典应用。

## 全排列

```javascript
// 有重复字符的全排列
function permute(s) {
  const res = [];
  const arr = s.split('').sort(); // 排序便于去重

  function backtrack(path, used) {
    if (path.length === s.length) {
      res.push(path);
      return;
    }
    for (let i = 0; i < arr.length; i++) {
      if (used[i]) continue;
      // 去重：相同字符中只取第一个未使用的
      if (i > 0 && arr[i] === arr[i - 1] && !used[i - 1]) continue;
      used[i] = true;
      backtrack(path + arr[i], used);
      used[i] = false;
    }
  }

  backtrack('', new Array(arr.length).fill(false));
  return res;
}

console.log(permute("aab")); // ["aab","aba","baa"]
console.log(permute("abc")); // ["abc","acb","bac","bca","cab","cba"]
```

## C++ 实现

```cpp
#include <vector>
#include <string>
#include <algorithm>
using namespace std;

void backtrack(string& s, vector<bool>& used, string& path,
               vector<string>& res) {
    if (path.size() == s.size()) {
        res.push_back(path);
        return;
    }
    for (int i = 0; i < s.size(); i++) {
        if (used[i]) continue;
        if (i > 0 && s[i] == s[i-1] && !used[i-1]) continue;
        used[i] = true;
        path.push_back(s[i]);
        backtrack(s, used, path, res);
        path.pop_back();
        used[i] = false;
    }
}

vector<string> permute(string s) {
    sort(s.begin(), s.end());
    vector<string> res;
    vector<bool> used(s.size(), false);
    string path;
    backtrack(s, used, path, res);
    return res;
}
```

## 组合

```javascript
// 字符串中取 k 个字符的所有组合
function combine(s, k) {
  const res = [];
  const arr = s.split('').sort();

  function backtrack(start, path) {
    if (path.length === k) {
      res.push(path);
      return;
    }
    for (let i = start; i < arr.length; i++) {
      if (i > start && arr[i] === arr[i - 1]) continue; // 去重
      backtrack(i + 1, path + arr[i]);
    }
  }

  backtrack(0, '');
  return res;
}

console.log(combine("abc", 2)); // ["ab","ac","bc"]
console.log(combine("aab", 2)); // ["aa","ab"]
```

## 子集

```javascript
// 字符串的所有子集（幂集）
function subsets(s) {
  const res = [];
  const arr = s.split('').sort();

  function backtrack(start, path) {
    res.push(path);
    for (let i = start; i < arr.length; i++) {
      if (i > start && arr[i] === arr[i - 1]) continue;
      backtrack(i + 1, path + arr[i]);
    }
  }

  backtrack(0, '');
  return res;
}

console.log(subsets("abc")); // ["","a","ab","abc","ac","b","bc","c"]
```

## 下一个排列

```javascript
// 字典序的下一个排列
function nextPermutation(s) {
  const arr = s.split('');
  let i = arr.length - 2;

  // 从右找第一个 arr[i] < arr[i+1]
  while (i >= 0 && arr[i] >= arr[i + 1]) i--;

  if (i >= 0) {
    // 从右找第一个大于 arr[i] 的元素
    let j = arr.length - 1;
    while (arr[j] <= arr[i]) j--;
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }

  // 反转 i+1 到末尾
  let l = i + 1, r = arr.length - 1;
  while (l < r) {
    [arr[l], arr[r]] = [arr[r], arr[l]];
    l++; r--;
  }

  return arr.join('');
}

console.log(nextPermutation("abc")); // "acb"
console.log(nextPermutation("cba")); // "abc" (回到最小)
```

## 复杂度分析

| 操作 | 时间 | 空间 |
|------|------|------|
| 全排列 | O(n!) | O(n) 递归栈 |
| 组合 | O(C(n,k)) | O(k) 递归栈 |
| 子集 | O(2^n) | O(n) 递归栈 |
| 下一个排列 | O(n) | O(1) |

## 关键技巧

1. **排序去重**：先排序，回溯时跳过重复字符
2. **used 数组**：排列中用 used 标记已使用元素
3. **start 参数**：组合/子集中用 start 控制起始位置
4. **剪枝优化**：剩余元素不足时提前返回

## 常见陷阱

1. **重复元素**：不排序不去重会产生重复结果
2. **引用传递**：path 用字符串则天然不可变，避免引用问题
3. **排序后操作**：某些题目要求返回原顺序的排列
4. **内存消耗**：n 较大时全排列结果数爆炸
