# Restore IP


```javascript
给定数字字符串，返回所有有效的IP地址组合。```

## 概念说明

给定一个只包含数字的字符串，将其分割为四段，每段在 0~255 范围内，组成有效的 IPv4 地址。合法条件：每段 1~3 位数字，不含前导零（"0" 合法，"01" 不合法），数值不超过 255。

## 核心思路

回溯时从当前位置尝试截取 1、2、3 个字符作为一段。剪枝条件：已有 4 段但字符串未用完则失败；前导零或超过 255 则跳过。递归参数：`start`（当前位置）和 `path`（已选段数）。实际上遍历空间很小（最多 3^3=27 种切法），暴力枚举即可。

## 复杂度分析

- **时间复杂度：** O(1)，实际最多 3^3=27 种分割方案（常数级）。
- **空间复杂度：** O(1)，递归栈深度固定为 4。

## 适用场景

- IP 地址格式验证
- 字符串分段枚举
- 带格式约束的分割问题

```
function restoreIpAddresses(s) {
  const res = [];
  function backtrack(start, path) {
    if (path.length === 4 && start === s.length) { res.push(path.join('.')); return; }
    if (path.length >= 4 || start >= s.length) return;
    for (let len = 1; len <= 3; len++) {
      if (start + len > s.length) break;
      const seg = s.slice(start, start+len);
      if ((seg.length > 1 && seg[0] === '0') || Number(seg) > 255) continue;
      path.push(seg);
      backtrack(start+len, path);
      path.pop();
    }
  }
  backtrack(0, []);
  return res;
}
console.log(restoreIpAddresses('25525511135'));
// ["255.255.11.135","255.255.111.35"]```


## 常见变体与技巧

- **迭代法：** 三层循环枚举三个分割点，代码更直观，适合面试手写。
- **IPv6 扩展：** 如需支持 IPv6，改为 8 段且每段 4 位十六进制数字。
- **提前终止：** 剩余字符数必须在 `[4-已选段数, 3*(4-已选段数)]` 范围内，否则提前返回。

  点击按钮查看结果
