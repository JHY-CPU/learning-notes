# String Encoding

### 什么是字符串编码

字符串编码是将一种表示形式转换为另一种的过程，用于数据压缩、传输安全和格式兼容。常见的编码方式包括 URL 编码、Base64、游程编码（RLE）等。

### 关键特性

- **URL 编码**：将特殊字符转为 %XX 格式，用于 HTTP 传输
- **Base64**：将二进制数据转为 ASCII 字符，用于邮件和数据嵌入
- **游程编码（RLE）**：将连续重复字符替换为"字符+次数"
- **哈夫曼编码**：根据字符频率构建最优前缀码

### 时间与空间复杂度

| 编码方式 | 编码时间 | 解码时间 | 压缩率 |
|---------|---------|---------|--------|
| RLE | O(n) | O(n) | 取决于重复度 |
| Base64 | O(n) | O(n) | 膨胀约 33% |
| URL 编码 | O(n) | O(n) | 膨胀（特殊字符变长） |
| 哈夫曼 | O(n log σ) | O(n) | 接近理论最优 |

### 适用场景 vs 替代方案

- **数据传输**：Base64 用于嵌入二进制到文本协议
- **压缩存储**：RLE 适合重复多的数据，通用场景用 gzip
- **URL 安全**：URL 编码确保特殊字符安全传输
- **高效压缩**：哈夫曼编码或 LZW 更通用

### 常见陷阱

- RLE 在重复少的数据上反而会膨胀（如 "abc" -> "1a1b1c"）
- Base64 编码后长度增加约 33%，不适合对大小敏感的场景
- 解码时未处理错误格式输入可能导致异常

```
// 游程编码
function runLengthEncode(s) {
  let res = '', count = 1;
  for (let i = 0; i < s.length; i++) {
    if (s[i] === s[i+1]) count++;
    else { res += count + s[i]; count = 1; }
  }
  return res;
}
function runLengthDecode(s) {
  let res = '';
  for (let i = 0; i < s.length; i+=2) {
    res += s[i+1].repeat(Number(s[i]));
  }
  return res;
}
console.log(runLengthEncode("AAABBBCCC")); // 3A3B3C
```


### 实际应用

- **HTTP 传输**：URL 中的中文和空格需要编码后才能传输
- **邮件附件**：MIME 使用 Base64 编码二进制文件
- **图像存储**：Data URL 中嵌入 Base64 编码的小图标
- **传真机**：使用 RLE 压缩黑白图像数据

  点击按钮查看结果
