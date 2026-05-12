# 53-字符串与数字转换 (String Number Conversion)

atoi（字符串转整数）和 itoa（整数转字符串）是字符串处理的基本操作。

## 字符串转整数（atoi）

```javascript
function myAtoi(s) {
  s = s.trim();
  if (!s) return 0;

  let i = 0, sign = 1;
  if (s[i] === '+' || s[i] === '-') {
    sign = s[i] === '-' ? -1 : 1;
    i++;
  }

  let num = 0;
  const INT_MAX = 2147483647;
  const INT_MIN = -2147483648;

  while (i < s.length && s[i] >= '0' && s[i] <= '9') {
    num = num * 10 + (s[i].charCodeAt(0) - 48);
    if (sign * num <= INT_MIN) return INT_MIN;
    if (sign * num >= INT_MAX) return INT_MAX;
    i++;
  }

  return sign * num;
}

console.log(myAtoi("   -42")); // -42
console.log(myAtoi("4193 with words")); // 4193
console.log(myAtoi("words and 987")); // 0
console.log(myAtoi("-91283472332")); // -2147483648
```

## C++ 实现

```cpp
#include <string>
#include <climits>
using namespace std;

int myAtoi(string s) {
    int i = 0, n = s.size();
    // 跳过空格
    while (i < n && s[i] == ' ') i++;
    // 符号
    int sign = 1;
    if (i < n && (s[i] == '+' || s[i] == '-')) {
        sign = s[i] == '-' ? -1 : 1;
        i++;
    }
    // 数字
    long num = 0;
    while (i < n && isdigit(s[i])) {
        num = num * 10 + (s[i] - '0');
        if (sign * num <= INT_MIN) return INT_MIN;
        if (sign * num >= INT_MAX) return INT_MAX;
        i++;
    }
    return (int)(sign * num);
}
```

## 整数转字符串

```javascript
function intToString(num) {
  if (num === 0) return '0';

  let negative = num < 0;
  num = Math.abs(num);
  let result = '';

  while (num > 0) {
    result = String(num % 10) + result;
    num = Math.floor(num / 10);
  }

  return negative ? '-' + result : result;
}

console.log(intToString(123));   // "123"
console.log(intToString(-456));  // "-456"
console.log(intToString(0));     // "0"
```

## 进制转换

```javascript
// 任意进制转十进制
function toDecimal(str, base) {
  let result = 0;
  for (const ch of str) {
    let digit;
    if (ch >= '0' && ch <= '9') digit = ch.charCodeAt(0) - 48;
    else digit = ch.charCodeAt(0) - 87; // 'a' = 10
    result = result * base + digit;
  }
  return result;
}

// 十进制转任意进制
function fromDecimal(num, base) {
  if (num === 0) return '0';
  const digits = '0123456789abcdef';
  let result = '';
  while (num > 0) {
    result = digits[num % base] + result;
    num = Math.floor(num / base);
  }
  return result;
}

console.log(toDecimal('ff', 16));   // 255
console.log(fromDecimal(255, 16)); // "ff"
console.log(fromDecimal(10, 2));   // "1010"
console.log(fromDecimal(255, 8));  // "377"
```

## 字符串转浮点数

```javascript
function myParseFloat(s) {
  s = s.trim();
  let i = 0, sign = 1;
  if (s[i] === '-') { sign = -1; i++; }
  else if (s[i] === '+') i++;

  let integer = 0;
  while (i < s.length && s[i] >= '0' && s[i] <= '9') {
    integer = integer * 10 + (s[i++] - 0);
  }

  let decimal = 0, divisor = 1;
  if (s[i] === '.') {
    i++;
    while (i < s.length && s[i] >= '0' && s[i] <= '9') {
      decimal = decimal * 10 + (s[i++] - 0);
      divisor *= 10;
    }
  }

  return sign * (integer + decimal / divisor);
}
```

## 常用内置方法

```javascript
// JS 内置方法
Number('42');       // 42
parseInt('42');     // 42
parseInt('1010', 2); // 10 (二进制)
parseFloat('3.14'); // 3.14
(42).toString();    // "42"
(255).toString(16); // "ff"
```

## 复杂度

| 操作 | 时间 | 空间 |
|------|------|------|
| atoi | O(n) | O(1) |
| itoa | O(log n) | O(log n) |
| 进制转换 | O(n) | O(log n) |

## 常见陷阱

1. **溢出检查**：转换过程中要检查是否超出整数范围
2. **前导空格**：atoi 需要先 trim
3. **正负号**：第一个非空字符可能是 + 或 -
4. **非数字字符**：遇到非数字应停止解析
5. **前导零**：进制转换时注意前导零的处理
