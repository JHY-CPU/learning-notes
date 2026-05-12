# 48-字符串反转 (String Reversal)

字符串反转是将字符串中字符的顺序颠倒，是最基础的字符串操作之一。

## 多种实现方式

```javascript
// 1. 内置方法
function reverseStr(s) {
  return s.split('').reverse().join('');
}

// 2. 反向遍历
function reverseStr2(s) {
  let r = '';
  for (let i = s.length - 1; i >= 0; i--) r += s[i];
  return r;
}

// 3. 双指针（需要转数组，JS 字符串不可变）
function reverseStr3(s) {
  const arr = s.split('');
  let l = 0, r = arr.length - 1;
  while (l < r) {
    [arr[l], arr[r]] = [arr[r], arr[l]];
    l++; r--;
  }
  return arr.join('');
}

// 4. 单词顺序反转
function reverseWords(s) {
  return s.trim().split(/\s+/).reverse().join(' ');
}

// 5. 反转字符串中的元音字母
function reverseVowels(s) {
  const vowels = new Set('aeiouAEIOU');
  const arr = s.split('');
  let l = 0, r = arr.length - 1;
  while (l < r) {
    while (l < r && !vowels.has(arr[l])) l++;
    while (l < r && !vowels.has(arr[r])) r--;
    [arr[l], arr[r]] = [arr[r], arr[l]];
    l++; r--;
  }
  return arr.join('');
}

console.log(reverseStr("algorithm")); // mhtirogla
console.log(reverseWords("hello world")); // world hello
console.log(reverseVowels("leetcode")); // leotcede
```

## C++ 实现

```cpp
#include <string>
#include <algorithm>
#include <sstream>
using namespace std;

// 原地反转（C++ string 可变）
void reverseStr(string& s) {
    int l = 0, r = s.size() - 1;
    while (l < r) swap(s[l++], s[r--]);
}

// 单词反转
string reverseWords(string s) {
    reverse(s.begin(), s.end());
    int n = s.size(), idx = 0;
    for (int i = 0; i < n; i++) {
        if (s[i] != ' ') {
            if (idx != 0) s[idx++] = ' ';
            int j = i;
            while (j < n && s[j] != ' ') s[idx++] = s[j++];
            reverse(s.begin() + idx - (j - i), s.begin() + idx);
            i = j;
        }
    }
    s.resize(idx);
    return s;
}

// 反转指定区间
void reverseRange(string& s, int l, int r) {
    while (l < r) swap(s[l++], s[r--]);
}

// 反转链表中的字符串值（模拟）
string reverseString(string s) {
    reverse(s.begin(), s.end());
    return s;
}
```

## K 连续字符反转

```javascript
// 每隔 k 个字符反转一组
function reverseStrK(s, k) {
  const arr = s.split('');
  for (let i = 0; i < arr.length; i += 2 * k) {
    let l = i, r = Math.min(i + k - 1, arr.length - 1);
    while (l < r) {
      [arr[l], arr[r]] = [arr[r], arr[l]];
      l++; r--;
    }
  }
  return arr.join('');
}
```

## 复杂度分析

| 方法 | 时间 | 空间 |
|------|------|------|
| split-reverse-join | O(n) | O(n) |
| 反向遍历拼接 | O(n) | O(n) |
| 双指针交换 | O(n) | O(n) 或 O(1)* |
| 递归反转 | O(n) | O(n) 栈 |

*C++ 中字符串可变，可以 O(1) 额外空间。

## 常见陷阱

1. **JavaScript 字符串不可变**：无法真正原地反转
2. **多余空格**：反转单词时要处理连续空格
3. **Unicode 字符**：emoji 等多字节字符反转会破坏
4. **边界条件**：空字符串、单字符的处理
