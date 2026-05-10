# 文本压缩工具 (Text Compression)

## 项目需求与功能分析

数据压缩是计算机科学中的经典问题。本项目实现 Huffman 编码和 LZ77 两种经典压缩算法，理解无损压缩的核心原理。

### 核心功能

- Huffman 编码压缩与解压
- LZ77 滑动窗口压缩与解压
- 压缩率计算与对比
- 频率统计与编码表展示
- 支持文件压缩与解压

### 应用场景

- 文件压缩（zip、gzip）
- 网络传输优化
- 数据库存储优化
- 多媒体编码基础

## 核心算法原理

### Huffman 编码

1. 统计每个字符的出现频率
2. 将每个字符作为叶子节点，频率作为权重
3. 每次取出权重最小的两个节点合并为新节点
4. 重复直到只剩一个根节点（Huffman 树）
5. 左路径编码为 0，右路径编码为 1

频率高的字符获得短编码，频率低的字符获得长编码，实现最优前缀编码。

### LZ77 算法

基于滑动窗口的字典压缩：
1. 维护一个滑动窗口（已处理的数据）
2. 在窗口中查找与当前位置匹配的最长子串
3. 输出三元组 (offset, length, next_char)
4. offset: 匹配距离窗口末尾的偏移
5. length: 匹配长度
6. next_char: 匹配后的下一个字符

## 完整代码实现

```python
import heapq
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional
import struct


class HuffmanNode:
    """Huffman 树节点"""
    def __init__(self, char=None, freq=0, left=None, right=None):
        self.char = char
        self.freq = freq
        self.left = left
        self.right = right

    def __lt__(self, other):
        return self.freq < other.freq

    def is_leaf(self):
        return self.char is not None


class HuffmanCoder:
    """Huffman 编码器"""

    def __init__(self):
        self.root: Optional[HuffmanNode] = None
        self.codes: Dict[str, str] = {}
        self.reverse_codes: Dict[str, str] = {}

    def build_tree(self, text: str) -> HuffmanNode:
        """构建 Huffman 树"""
        freq = Counter(text)
        if not freq:
            return None

        # 特殊情况：只有一个字符
        if len(freq) == 1:
            char = list(freq.keys())[0]
            self.root = HuffmanNode(char=char, freq=freq[char])
            self.codes = {char: '0'}
            self.reverse_codes = {'0': char}
            return self.root

        heap = [HuffmanNode(char=c, freq=f) for c, f in freq.items()]
        heapq.heapify(heap)

        while len(heap) > 1:
            left = heapq.heappop(heap)
            right = heapq.heappop(heap)
            merged = HuffmanNode(freq=left.freq + right.freq, left=left, right=right)
            heapq.heappush(heap, merged)

        self.root = heap[0]
        self._build_codes(self.root, "")
        return self.root

    def _build_codes(self, node: HuffmanNode, code: str):
        if node is None:
            return
        if node.is_leaf():
            self.codes[node.char] = code if code else "0"
            self.reverse_codes[code if code else "0"] = node.char
            return
        self._build_codes(node.left, code + "0")
        self._build_codes(node.right, code + "1")

    def encode(self, text: str) -> str:
        """编码"""
        if not self.root:
            self.build_tree(text)
        return ''.join(self.codes[c] for c in text)

    def decode(self, encoded: str) -> str:
        """解码"""
        if not self.root:
            return ""
        result = []
        node = self.root

        # 单字符特殊情况
        if node.is_leaf():
            return node.char * len(encoded)

        for bit in encoded:
            node = node.left if bit == '0' else node.right
            if node.is_leaf():
                result.append(node.char)
                node = self.root
        return ''.join(result)

    def get_frequency_table(self, text: str) -> Dict[str, int]:
        return Counter(text)

    def compression_ratio(self, original: str, encoded: str) -> float:
        original_bits = len(original) * 8
        compressed_bits = len(encoded)
        return 1 - compressed_bits / original_bits if original_bits > 0 else 0


class LZ77Compressor:
    """LZ77 压缩器"""

    def __init__(self, window_size=4096, lookahead_size=18):
        self.window_size = window_size
        self.lookahead_size = lookahead_size

    def compress(self, text: str) -> List[Tuple[int, int, str]]:
        """压缩为三元组列表"""
        tokens = []
        pos = 0

        while pos < len(text):
            window_start = max(0, pos - self.window_size)
            window = text[window_start:pos]
            lookahead = text[pos:pos + self.lookahead_size]

            best_offset = 0
            best_length = 0

            # 在滑动窗口中找最长匹配
            for length in range(min(len(lookahead), self.lookahead_size), 0, -1):
                substring = lookahead[:length]
                idx = window.rfind(substring)
                if idx != -1:
                    best_offset = len(window) - idx
                    best_length = length
                    break

            if best_length > 0:
                next_char = text[pos + best_length] if pos + best_length < len(text) else ''
                tokens.append((best_offset, best_length, next_char))
                pos += best_length + 1
            else:
                tokens.append((0, 0, text[pos]))
                pos += 1

        return tokens

    def decompress(self, tokens: List[Tuple[int, int, str]]) -> str:
        """解压"""
        result = []
        for offset, length, char in tokens:
            if offset > 0:
                start = len(result) - offset
                for i in range(length):
                    result.append(result[start + i])
            if char:
                result.append(char)
        return ''.join(result)

    def compression_ratio(self, original: str, tokens: List) -> float:
        # 每个三元组约 3 字节 (offset 12bit, length 4bit, char 8bit)
        compressed_bits = len(tokens) * 24
        original_bits = len(original) * 8
        return 1 - compressed_bits / original_bits if original_bits > 0 else 0


def compare_algorithms(text: str):
    """对比两种压缩算法"""
    print(f"原文长度: {len(text)} 字符 ({len(text)*8} bits)")
    print("=" * 50)

    # Huffman
    huff = HuffmanCoder()
    huff.build_tree(text)
    encoded = huff.encode(text)
    decoded = huff.decode(encoded)
    ratio_h = huff.compression_ratio(text, encoded)

    print(f"\nHuffman 编码:")
    print(f"  编码长度: {len(encoded)} bits")
    print(f"  压缩率: {ratio_h:.2%}")
    print(f"  编码表 (前10个):")
    for i, (char, code) in enumerate(sorted(huff.codes.items(), key=lambda x: len(x[1]))[:10]):
        display = repr(char) if char in ('\n', '\t', ' ') else char
        print(f"    {display:>6} -> {code}")
    print(f"  解压正确: {decoded == text}")

    # LZ77
    lz = LZ77Compressor()
    tokens = lz.compress(text)
    decompressed = lz.decompress(tokens)
    ratio_l = lz.compression_ratio(text, tokens)

    print(f"\nLZ77 压缩:")
    print(f"  三元组数: {len(tokens)}")
    print(f"  压缩率: {ratio_l:.2%}")
    print(f"  解压正确: {decompressed == text}")

    print(f"\n{'='*50}")
    print(f"{'算法':<12} {'压缩率':<12} {'正确性':<8}")
    print(f"{'Huffman':<12} {ratio_h:<12.2%} {'OK' if decoded==text else 'FAIL':<8}")
    print(f"{'LZ77':<12} {ratio_l:<12.2%} {'OK' if decompressed==text else 'FAIL':<8}")
```

## 测试用例

```python
import unittest

class TestCompression(unittest.TestCase):
    def test_huffman_roundtrip(self):
        text = "hello world! hello compression!"
        h = HuffmanCoder()
        h.build_tree(text)
        encoded = h.encode(text)
        decoded = h.decode(encoded)
        self.assertEqual(decoded, text)

    def test_huffman_single_char(self):
        h = HuffmanCoder()
        h.build_tree("aaaa")
        self.assertEqual(h.decode(h.encode("aaaa")), "aaaa")

    def test_lz77_roundtrip(self):
        text = "ABABABABABABC"
        lz = LZ77Compressor()
        tokens = lz.compress(text)
        self.assertEqual(lz.decompress(tokens), text)

    def test_lz77_repetitive(self):
        text = "a" * 100
        lz = LZ77Compressor()
        tokens = lz.compress(text)
        self.assertLess(len(tokens), 20)  # 重复数据压缩效果好

    def test_huffman_compression_ratio(self):
        text = "aaabbbccc"
        h = HuffmanCoder()
        h.build_tree(text)
        encoded = h.encode(text)
        self.assertGreater(h.compression_ratio(text, encoded), 0)

    def test_empty_string(self):
        h = HuffmanCoder()
        self.assertEqual(h.encode(""), "")
        self.assertEqual(h.decode(""), "")

if __name__ == '__main__':
    unittest.main()
```

## 扩展方向

1. **LZ78 / LZW**：实现更高效的字典压缩算法
2. **DEFLATE**：结合 LZ77 + Huffman（gzip 使用的算法）
3. **算术编码**：比 Huffman 更接近信息熵的编码方式
4. **Burrows-Wheeler 变换**：bzip2 使用的预处理变换
5. **二进制文件支持**：处理任意二进制数据
6. **压缩率自适应**：根据数据特征选择最佳算法
7. **分块压缩**：支持大文件的分块压缩与解压
