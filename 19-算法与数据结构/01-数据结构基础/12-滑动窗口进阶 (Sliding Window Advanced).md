## 12-滑动窗口进阶 (Sliding Window Advanced)

进阶滑动窗口技巧，包括多条件窗口、双哈希表维护、计数窗口等复杂场景。

## 复杂窗口场景

```javascript

// 1. 字符串排列判断
// s2 中是否包含 s1 的排列
function checkInclusion(s1, s2) {
  let need = {}, window = {};
  for (let c of s1) need[c] = (need[c] || 0) + 1;

  let left = 0, right = 0, valid = 0;

  while (right < s2.length) {
    let c = s2[right];
    right++;

    if (need[c]) {
      window[c] = (window[c] || 0) + 1;
      if (window[c] === need[c]) valid++;
    }

    // 窗口大小等于 s1 长度时收缩
    while (right - left >= s1.length) {
      if (valid === Object.keys(need).length) return true;

      let d = s2[left];
      left++;
      if (need[d]) {
        if (window[d] === need[d]) valid--;
        window[d]--;
      }
    }
  }
  return false;
}

// 2. 找到字符串中所有字母异位词
function findAnagrams(s, p) {
  let need = {}, window = {};
  for (let c of p) need[c] = (need[c] || 0) + 1;

  let left = 0, right = 0, valid = 0;
  let result = [];

  while (right < s.length) {
    let c = s[right]; right++;
    if (need[c]) {
      window[c] = (window[c] || 0) + 1;
      if (window[c] === need[c]) valid++;
    }

    while (right - left >= p.length) {
      if (valid === Object.keys(need).length) result.push(left);
      let d = s[left]; left++;
      if (need[d]) {
        if (window[d] === need[d]) valid--;
        window[d]--;
      }
    }
  }
  return result;
}
```

## 交互演示
